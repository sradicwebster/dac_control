import numpy as np
from typing import Optional

class Battery:
    """
    Battery model
    """
    def __init__(self,
                 capacity: int,
                 power_max: int,
                 charge_eff: float,
                 discharge_eff: float,
                 soc_min: float,
                 soc_max: float,
                 dt: int,
                 ) -> None:
        """

        Args:
            capacity (int): battery capacity (kWh)
            power_max (int): maximum power for charging or discharging (kW)
            charge_eff (float): charging efficiency
            discharge_eff (float): discharging efficiency
            soc_min (float): minimum state of charge as a fraction of capacity
            soc_max (float): maximum state of charge as a fraction of capacity
            dt (int): time step (min)

        """
        self.capacity = capacity
        self.power_max = power_max
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff
        self.soc_min = soc_min * self.capacity
        self.soc_max = soc_max * self.capacity
        self.soc = None
        self.dt = dt

    def reset(self,
              n: Optional[int] = 1,
              ) -> np.ndarray:
        """ Sets the battery to fully charged

        Args:
            n (int): number of parallel experiments

        Returns:
            (np.ndarray): battery SOC as a fraction of maximum capacity in an array n x 1

        """
        self.soc = np.ones((n, 1)) * self.soc_max
        if self.capacity == 0:
            return self.soc
        else:
            return self.soc / self.soc_max

    def step(self,
             power: np.ndarray,
             ) -> np.ndarray:
        """ Updates the battery SOC based on power (positive is discharging and negative is
            charging)

        Args:
            power (np.ndarray): battery charge or discharge power

        Returns:
            (np.ndarray): battery SOC as a fraction of maximum capacity in an array n x 1

        """
        if power.ndim == 1:
            power = power.reshape(1, -1)
        power = np.clip(power, -self.power_max, self.power_max)
        energy = np.where(power > 0,
                          power * (self.dt / 60) / self.discharge_eff,
                          power * (self.dt / 60) * self.charge_eff)
        self.soc = np.clip(self.soc - energy, self.soc_min, self.soc_max)
        if self.capacity == 0:
            return self.soc
        else:
            return self.soc / self.soc_max

    def discharge_power(self,
                        ) -> np.ndarray:
        """ Available discharge power

        Returns:
            (np.ndarray): power available in an array n x 1

        """
        discharge = np.minimum((self.soc - self.soc_min) * self.capacity / (self.dt / 60),
                               np.array([self.power_max]))
        return discharge
