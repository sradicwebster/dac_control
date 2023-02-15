import numpy as np


class Battery:
    """
    Battery model
    """
    def __init__(self,
                 capacity: float,
                 power_max: float,
                 charge_eff: float,
                 discharge_eff: float,
                 soc_min: float,
                 soc_max: float,
                 dt: float,
    ):
        """

        Args:
            capacity: battery capacity (kWh)
            power_max: maximum power for charging or discharging (kW)
            charge_eff: charging efficiency
            discharge_eff: discharging efficiency
            soc_min: minimum state of charge as a fraction of capacity
            soc_max: maximum state of charge as a fraction of capacity
            dt: time step (min)
        """
        self.capacity = capacity
        self.power_max = power_max
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff
        self.soc_min = soc_min * capacity
        self.soc_max = soc_max * capacity
        self.soc = np.array([self.soc_max])
        self.dt = dt

    def reset(self):
        self.soc = np.array([self.soc_max])
        return self.soc

    def step(self, power):
        """

        Args:
            power: power to battery (positive = discharging, negative = charging)

        Returns:

        """
        power = np.clip(power, -self.power_max, self.power_max)
        energy = np.where(power > 0,
                          power * (self.dt / 60) / self.discharge_eff,
                          power * (self.dt / 60) * self.charge_eff)
        self.soc = np.clip(self.soc - energy, self.soc_min, self.soc_max)
        return self.soc

    def charge_power(self):
        charge = max((self.soc - self.soc_max) / (self.dt / 60), -np.array([self.power_max]))
        return charge

    def discharge_power(self):
        discharge = min((self.soc - self.soc_min) / (self.dt / 60), np.array([self.power_max]))
        return discharge
