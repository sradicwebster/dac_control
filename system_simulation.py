from tqdm import tqdm
import numpy as np
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from copy import deepcopy

from wind_data_processing import load_wind_data

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("t90_to_k", lambda t90: np.log(1 / 0.1) / (60 * t90))


@hydra.main(version_base=None, config_path="configs", config_name="main")
def run(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)

    with wandb.init(project="dac_system", config=OmegaConf.to_object(cfg)) as _:
        wandb.config.update({"kinetics_": cfg.kinetics._target_.split(".")[-1],
                             "sizing_": cfg.sizing._target_.split(".")[-1],
                             "controller_": cfg.controller._target_.split(".")[-1],
                             })

        wind_power_series = load_wind_data(cfg.wind.file, cfg.dt, cfg.wind.var)
        wind_max = np.max(wind_power_series)
        if cfg.wind.max_only:
            wind_power_series = wind_max * np.ones_like(wind_power_series)

        iter_per_hour = 60 / cfg.dt
        iters = int(cfg.T * iter_per_hour)
        assert iters < len(wind_power_series)

        dac = instantiate(cfg.dac,
                          process_conditions=cfg.process_conditions,
                          sizing_cfg=cfg.sizing,
                          kinetics_cfg=cfg.kinetics,
                          _recursive_=False)
        battery = hydra.utils.instantiate(cfg.battery)

        if "dynamics_model" in cfg:
            dynamics_model = hydra.utils.instantiate(cfg.dynamics_model,
                                                     dac=deepcopy(dac),
                                                     battery=deepcopy(battery),
                                                     wind_power=wind_power_series,
                                                     _recursive_=False)
            controller = instantiate(cfg.controller, model=dynamics_model)
            wandb.config.update({"dynamics_model_": cfg.dynamics_model._target_.split(".")[-1],
                                 "wind_model_": cfg.dynamics_model.wind_model._target_.split(".")[-1],
                                 })
        else:
            controller = instantiate(cfg.controller, _recursive_=False)

        state = np.concatenate((wind_power_series[0] / wind_max,
                                battery.reset().flatten(),
                                dac.reset().flatten(),
                                np.zeros(dac.num_units),
                                ))

        hour = 0
        total_co2_captured = 0
        wandb.config.CO2_per_cycle_kg = dac.num_units * (dac.q_CO2_eq["ad"] - dac.q_CO2_eq["de"])
        if "geometry" in cfg.sizing:
            wandb.config.sizing.update({"geometry": {"volume": dac.sizing.geometry.volume}})
        for u in range(cfg.dac.num_units):
            wandb.log({f"DAC_{u + 1}_loading_(kg)": state[2 + u] * dac.q_CO2_eq["ad"],
                       "Time_(h)": hour,
                       }, commit=False)
        wandb.log({"Wind_power_(kW)": state[0] * wind_max,
                   "Battery_SOC_(kWh)": state[1] * battery.capacity,
                   "Time_(h)": hour,
                   })

        for i in tqdm(range(iters)):

            controls = controller.policy(state)
            dac_power = dac.step(controls, update_state=False, return_power=True)
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
                dac_power = dac.step(controls, update_state=False, return_power=True)
                power_deficit = dac_power - wind_power - battery_discharge

            battery_power = dac_power - wind_power

            state = np.concatenate((wind_power / wind_max,
                                    battery.step(battery_power).flatten(),
                                    dac.step(controls).flatten(),
                                    controls,
                                    ))

            if (i + 1) % iter_per_hour == 0:
                hour += 1
            for u in range(cfg.dac.num_units):
                wandb.log({f"DAC_{u + 1}_loading_(kg)": state[2 + u] * dac.q_CO2_eq["ad"],
                           "Time_(h)": hour,
                           }, commit=False)
            wandb.log({"Wind_power_(kW)": state[0] * wind_max,
                       "DAC_power_(kW)": dac_power,
                       "Battery_SOC_(kWh)": state[1] * battery.capacity,
                       "CO2_captured_(kg)": dac.CO2_captured,
                       "Time_(h)": hour,
                       })
            total_co2_captured += dac.CO2_captured

        wandb.log({"CO2_rate_(kg/h)": total_co2_captured / cfg.T,
                   "CO2_rate_(ton/yr)": total_co2_captured / cfg.T / 1e3 * 24 * 365,
                   })
        if "geometry" in cfg.sizing:
            wandb.log({"Productivity_(kg/h/m^3)": total_co2_captured / cfg.T /
                                       cfg.sizing.geometry.volume})


if __name__ == "__main__":
    run()
