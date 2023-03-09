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
                             "controller_": cfg.controller._target_.split(".")[-1]})
        wind_power_series = load_wind_data("wind_power_sim", cfg.dt, cfg.get("wind_var", 0))
        wind_max = np.max(wind_power_series)
        if cfg.get("wind_max_only", False):
            wind_power_series = wind_max * np.ones_like(wind_power_series)

        iter_per_hour = 60 / cfg.dt
        if cfg.T != -1:
            iters = int(cfg.T * iter_per_hour)
        else:
            iters = len(wind_power_series) - 1
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

        prev_controls = np.zeros(dac.num_units)
        state = np.concatenate((wind_power_series[0] / wind_max,
                                battery.reset().flatten(),
                                dac.reset().flatten(),
                                prev_controls))

        hour = 0
        wind_utilisation = 0
        total_co2_captured = 0
        desorptions = 0
        wandb.config.CO2_per_cycle_kg = dac.num_units * (dac.m_CO2_eq["ad"] - dac.m_CO2_eq["de"])
        if "geometry" in cfg.sizing:
            wandb.config.sizing.update({"geometry": {"volume": dac.sizing.geometry.volume}})
        for u in range(cfg.dac.num_units):
            wandb.log({f"DAC_{u + 1}_loading_(kg)": state[2 + u] * dac.m_CO2_eq["ad"],
                       "Time_(h)": hour,
                       }, commit=False)
        wandb.log({"Wind_power_(kW)": state[0] * wind_max,
                   "Battery_SOC_(kWh)": state[1] * battery.capacity,
                   "Time_(h)": hour})

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

            battery_power = np.clip(dac_power - wind_power, -battery.power_max, battery.power_max)

            state = np.concatenate((wind_power / wind_max,
                                    battery.step(battery_power).flatten(),
                                    dac.step(controls).flatten(),
                                    controls))

            if (i + 1) % iter_per_hour == 0:
                hour += 1
            for u in range(cfg.dac.num_units):
                wandb.log({f"DAC_{u + 1}_loading_(kg)": state[2 + u] * dac.m_CO2_eq["ad"],
                           "Time_(h)": hour,
                           }, commit=False)
            wandb.log({"Wind_power_(kW)": state[0] * wind_max,
                       "DAC_power_(kW)": dac_power,
                       "Battery_SOC_(kWh)": state[1] * battery.capacity,
                       "CO2_captured_(kg)": dac.CO2_captured,
                       "Time_(h)": hour})
            total_co2_captured += dac.CO2_captured
            if wind_power != 0:
                wind_utilisation += (dac_power - battery_power) / wind_power
            desorptions += np.logical_and(prev_controls != -1, controls == -1).sum()
            prev_controls = controls

        co2_per_hour = total_co2_captured / (iters / iter_per_hour)
        wandb.log({"Performance":
                       {"CO2_rate_(kg/h)": co2_per_hour,
                        "CO2_rate_(ton/yr)": co2_per_hour / 1e3 * 24 * 365,
                        "wind_utilisation": wind_utilisation / iters,
                        "desorption_rate": desorptions / (dac.num_units * iters / iter_per_hour)},
                   })
        if "geometry" in cfg.sizing:
            wandb.log({"Performance":
                           {"Productivity_(kg/h/m^3)": co2_per_hour / dac.sizing.geometry.volume},
                       })


if __name__ == "__main__":
    run()
