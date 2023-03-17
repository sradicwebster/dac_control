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
        self.prev_controls = None

    def reset(self,
              state: np.ndarray,
              n: int = 1,
              ):
        """

        Args:
            state:
            n:

        Returns:

        """
        self.dac.m_CO2 = np.repeat(state[2: 2+self.num_units].reshape(1, -1) *
                                   self.dac.m_CO2_eq["ad"], n, axis=0)
        self.dac.m_H2O = np.repeat(state[2+self.num_units: 2+2*self.num_units].reshape(1, -1) *
                                   self.dac.m_H2O_eq["ad"], n, axis=0)
        self.dac.T_units = np.repeat(state[2+2*self.num_units: 2+3*self.num_units].reshape(1, -1) *
                                     self.dac.process_conditions.T_de, n, axis=0)
        self.battery.soc = np.repeat(state[1].reshape(1, -1) * self.battery.soc_max, n, axis=0)

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
