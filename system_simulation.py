import os
from tqdm import tqdm
import numpy as np
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from controllers import *
from wind_data_processing import load_wind_data


OmegaConf.register_new_resolver("t90_to_k", lambda t90: np.log(1 / 0.1) / (60 * t90))


@hydra.main(version_base=None, config_path="configs", config_name="main")
def run(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)

    with wandb.init(project='dac_system', config=omegaconf.OmegaConf.to_object(cfg)) as _:

        wind_power_series = load_wind_data(cfg.dt)
        dac = instantiate(cfg.dac,
                          process_conditions=cfg.process_conditions,
                          unit_sizing_cfg=cfg.unit_sizing,
                          kinetics_cfg=cfg.kinetics,
                          _recursive_=False)
        battery = hydra.utils.instantiate(cfg.battery)

        # TODO make dynamics model instantiation neater
        dynamics_model = hydra.utils.instantiate(cfg.dynamics_model, _recursive_=False)
        # dynamics_model.dac = hydra.utils.instantiate(dynamics_model.dac)
        # dynamics_model.battery = hydra.utils.instantiate(dynamics_model.battery)
        # dynamics_model.wind_model = hydra.utils.instantiate(dynamics_model.wind_model,
        # wind_power_series)

        controller = hydra.utils.instantiate(cfg.controller, model=dynamics_model)

        state = np.concatenate((wind_power_series[0],
                                battery.reset().flatten(),
                                dac.reset().flatten(),
                                np.zeros(dac.num_units),
                                ))
        hour = 0
        wandb.log({"wind_power": state[0], "battery_soc": state[1], "time (h)": hour})
        for i in range(cfg.dac.num_units):
            wandb.log({f"dac_{i + 1}_loading": state[2 + i],
                       "time (h)": hour})

        iter_per_hour = 60 / cfg.dt
        iters = int(cfg.T * iter_per_hour)
        total_co2_captured = 0
        for i in tqdm(range(iters)):

            controls = controller.policy(state)
            dac_power = dac.power_requirement(controls)
            wind_power = wind_power_series[i + 1]

            battery_discharge = battery.discharge_power()
            power_deficit = dac_power - wind_power - battery_discharge
            unit = 0
            mode = 1
            while power_deficit > 1e-3:
                if controls[unit] == mode:
                    controls[unit] = 0
                unit += 1
                if unit == dac.num_units:
                    unit = 0
                    mode = -1
                dac_power = dac.power_requirement(controls)
                power_deficit = dac_power - wind_power - battery_discharge

            battery_power = dac_power - wind_power
            state = np.concatenate((wind_power,
                                    battery.step(battery_power).flatten(),
                                    dac.step(controls).flatten(),
                                    controls
                                    ))

            if (i + 1) % iter_per_hour == 0:
                hour += 1
            for u in range(cfg.dac.num_units):
                wandb.log({f"dac_{u + 1}_loading": state[2 + u],
                           "time (h)": hour},
                          commit=False)
            wandb.log({"wind_power": state[0],
                       "battery_soc": state[1],
                       "co2_captured": dac.CO2_captured,
                       "time (h)": hour})
            total_co2_captured += dac.CO2_captured

        wandb.log({"co2_rate": total_co2_captured / cfg.T})


if __name__ == "__main__":
    run()
