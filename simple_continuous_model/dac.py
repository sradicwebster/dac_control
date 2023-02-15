import numpy as np


class DirectAirCapture:
    """
    Direct air capture model with continuous control
    """

    def __init__(self,
                 num_units: int,
                 adsorbent_capacity: float,
                 adsorption_power: float,
                 desorption_power: float,
                 adsorption_rate: float,
                 desorption_rate: float,
                 dt: float,
                 ):
        """

        Args:
            num_units: number of DAC units
            adsorbent_capacity: adsorbent CO2 capacity per unit (kg)
            adsorption_power: adsorption maximum power per unit (kW)
            desorption_power: desorption maximum power per unit (kW)
            adsorption_rate: rate of CO2 adsorption (kg / kWh)
            desorption_rate: rate of CO2 desorption (kg / kWh)
            dt: time step (min)
        """
        self.num_units = num_units
        self.adsorbent_capacity = adsorbent_capacity
        self.adsorption_power = adsorption_power
        self.desorption_power = desorption_power
        self.adsorption_rate = adsorption_rate
        self.desorption_rate = desorption_rate
        self.dt = dt
        self.capacity = np.zeros(num_units)

    def reset(self):
        self.capacity = np.zeros(self.num_units)
        return self.capacity

    def step(self, power: np.ndarray):
        """

        Args:
            power: power to each unit (kW). Note: positive power is adsorption and negative power
             is desorption

        Returns:

        """
        power = np.clip(power, -self.desorption_power, self.adsorption_power)
        adsorbed = np.where(power > 0,
                            self.adsorption_rate * power * self.dt / 60,
                            self.desorption_rate * power * self.dt / 60)
        self.capacity = np.clip(self.capacity + adsorbed, 0, self.adsorbent_capacity)
        return self.capacity
