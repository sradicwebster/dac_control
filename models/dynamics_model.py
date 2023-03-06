import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate

from models.dac import DAC
from models.battery import Battery


class KnownModel:
    """
    Provide a perfect model (apart from wind prediction) for MPC style controller
    """
    def __init__(self,
                 dac: DAC,
                 battery: Battery,
                 wind_model: DictConfig,
                 wind_power: np.ndarray
                 ) -> None:
        """

        Args:
            dac (DAC): initialised DAC instance
            battery (Battery): initialised Battery instance
            wind_model (DictConfig): wind model parameters
            wind_power (np.ndarray): wind power series (kW)
        """
        self.dac = dac
        self.battery = battery
        self.wind_model = instantiate(wind_model, wind_power=wind_power)
        self.num_units = dac.num_units

    def step(self,
             state: np.ndarray,
             controls: np.ndarray,
             ) -> np.ndarray:
        """ Predict the next state for a given control action

        Args:
            state (np.ndarray): current state of the system
            controls (np.ndarray): DAC controls (mode)

        Returns:
            (np.ndarray): next state

        """
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
