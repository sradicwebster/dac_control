import numpy as np

#from simple_discrete_model.dac.dac import DirectAirCapture
from battery import Battery
from wind_model import BaseModel


class BaseDynamics:
    def __init__(self,
                 dac,
                 battery: Battery,
                 wind_model: BaseModel,
                 ):
        self.dac = dac
        self.battery = battery
        self.wind_model = wind_model
        self.num_units = dac.num_units

    def step(self,
             state: np.ndarray,
             controls: np.ndarray,
             ):
        pass


class KnownModel(BaseDynamics):
    def __init__(self,
                 dac,
                 battery: Battery,
                 wind_model: BaseModel):

        super().__init__(dac,
                         battery,
                         wind_model)

    def step(self,
             state: np.ndarray,
             controls: np.ndarray,
             ):
        dac_power = self.dac.power_requirement(controls)
        wind_power = self.wind_model.next(state)
        battery_discharge = self.battery.discharge_power()
        power_deficit = (dac_power - wind_power - battery_discharge) > 1e-3

        if power_deficit.any():
            for mode in [1, -1]:
                for unit in range(self.num_units):
                    controls[:, unit] *= np.logical_not(np.logical_and(power_deficit.flatten(),
                                                                       controls[:, unit] == mode))
                    dac_power = self.dac.power_requirement(controls)
                    power_deficit = (dac_power - wind_power - battery_discharge) > 1e-3

        battery_power = dac_power - wind_power
        next_state = np.concatenate((wind_power,
                                     self.battery.step(battery_power),
                                     self.dac.step(controls),
                                     controls,
                                     ), axis=1)
        return next_state
