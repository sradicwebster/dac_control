import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf, DictConfig
import wandb
from typing import Tuple
import gym
from gym.spaces import Box, MultiDiscrete
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from wind_data_processing import load_wind_data

OmegaConf.register_new_resolver("t90_to_k", lambda t90: np.log(1 / 0.1) / (60 * t90))


class DACEnv(gym.Env):
    def __init__(self,
                 cfg: DictConfig,
                 ):
        super().__init__()
        self.cfg = cfg
        self.wind_power_series = load_wind_data(cfg.dt)
        self.wind_max = self.wind_power_series.max()
        self.dac = instantiate(cfg.dac,
                               process_conditions=cfg.process_conditions,
                               unit_sizing_cfg=cfg.unit_sizing,
                               kinetics_cfg=cfg.kinetics,
                               _recursive_=False)
        self.battery = hydra.utils.instantiate(cfg.battery)

        self.action_space = MultiDiscrete(3 * np.ones(self.dac.num_units))
        obs_dim = 2 * self.dac.num_units + 2
        obs_low = np.concatenate((np.zeros(self.dac.num_units + 2), -np.ones(self.dac.num_units)))
        obs_high = np.ones(obs_dim)
        self.observation_space = Box(obs_low, obs_high, shape=(obs_dim,), dtype=np.float64)

        self.state = None
        self.dac_power = None
        self.i = None

    def reset(self) -> np.ndarray:
        self.i = 0
        self.state = np.concatenate((self.wind_power_series[self.i] / self.wind_max,
                                     self.battery.reset().flatten(),
                                     self.dac.reset().flatten(),
                                     np.ones(self.dac.num_units),
                                     ))
        return self.state

    def step(self,
             action: np.ndarray,
             ) -> Tuple[np.ndarray, float, bool, dict]:
        controls = action - 1
        self.dac_power = self.dac.step(controls, update_state=False, return_power=True)
        wind_power = self.wind_power_series[self.i + 1]

        battery_discharge = self.battery.discharge_power()
        power_deficit = self.dac_power - wind_power - battery_discharge
        unit = 0
        mode = 1
        while power_deficit > 1e-3:
            if controls[unit] == mode:
                controls[unit] = 0
            unit += 1
            if unit == self.dac.num_units:
                unit = 0
                mode = -1
            dac_power = self.dac.step(controls, update_state=False, return_power=True)
            power_deficit = dac_power - wind_power - battery_discharge

        battery_power = self.dac_power - wind_power
        prev_controls = self.state[self.dac.num_units + 2:]
        self.state = np.concatenate((wind_power / self.wind_max,
                                     self.battery.step(battery_power).flatten(),
                                     self.dac.step(controls).flatten(),
                                     controls,
                                     ))
        start_desorb = np.logical_and(prev_controls != -1, controls == -1).sum()
        reward = self.dac.CO2_captured.item() / 1e3 - self.cfg.desorb_pen * start_desorb

        self.i += 1
        done = False
        if self.i == len(self.wind_power_series)-1:
            done = True

        return self.state, reward, done, {}


class WandBLogging(BaseCallback):
    def __init__(self,
                 verbose: int = 0,
                 ):
        super(WandBLogging, self).__init__(verbose)

    def _on_training_start(self) -> None:
        self.env = self.training_env.envs[0].env

    def _on_step(self) -> bool:
        for u in range(self.env.dac.num_units):
            wandb.log({f"dac_{u + 1}_loading": self.env.state[2 + u] * self.env.dac.q_CO2_eq["ad"]},
                      commit=False)
        wandb.log({"wind_power": self.env.state[0] * self.env.wind_max,
                   "dac_power": self.env.dac_power,
                   "battery_soc": self.env.state[1] * self.env.battery.capacity,
                   "co2_captured": self.env.dac.CO2_captured})
        return True


@hydra.main(version_base=None, config_path="configs", config_name="main")
def run(cfg: DictConfig):
    wandb.init(project="dac_system_rl_train", config=OmegaConf.to_object(cfg),
               sync_tensorboard=True)
    env = DACEnv(cfg)
    # env = make_vec_env(DACEnv, n_envs=4, env_kwargs={"cfg": cfg})
    agent = instantiate(cfg.controller.algorithm, env=env, verbose=0, tensorboard_log="tb_log")
    agent.learn(cfg.controller.training_timesteps,
                callback=WandBLogging(verbose=0))
    agent.save(f"trained_agents/{wandb.run.name}")


if __name__ == "__main__":
    run()
