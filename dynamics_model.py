import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate

from dac_models.dac import DAC
from battery import Battery


class KnownModel:
    def __init__(self,
                 dac: DAC,
                 battery: Battery,
                 wind_model: DictConfig,
                 wind_power: np.ndarray
                 ):
        self.dac = dac
        self.battery = battery
        self.wind_model = instantiate(wind_model, wind_power=wind_power)
        self.num_units = dac.num_units

    def step(self,
             state: np.ndarray,
             controls: np.ndarray,
             ):
        dac_power = self.dac.step(controls, update_state=False, return_power=True)
        wind_power = self.wind_model.next(state)
        battery_discharge = self.battery.discharge_power()
        power_deficit = (dac_power - wind_power - battery_discharge) > 1e-3

        if power_deficit.any():
            for mode in [1, -1]:
                for unit in range(self.num_units):
                    controls[:, unit] *= np.logical_not(np.logical_and(power_deficit.flatten(),
                                                                       controls[:, unit] == mode))
                    dac_power = self.dac.step(controls, update_state=False, return_power=True)
                    power_deficit = (dac_power - wind_power - battery_discharge) > 1e-3

        battery_power = dac_power - wind_power
        next_state = np.concatenate((wind_power,
                                     self.battery.step(battery_power),
                                     self.dac.step(controls),
                                     controls,
                                     ), axis=1)
        return next_state
