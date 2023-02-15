import numpy as np


class RuleMaxCapacity:
    def __init__(self,
                 state_constants,
                 dt,
                 num_units,
                 capacity_limit_low,
                 capacity_limit_high,
                 uncertainty_adjustment=0,
                 ):
        self.max_capacity = state_constants[0]
        self.adsorption_power = state_constants[1]
        self.desorption_power = state_constants[2]
        self.soc_min = state_constants[3]
        self.soc_max = state_constants[4]
        self.power_max = state_constants[5]
        self.dt = dt
        self.num_units = num_units
        self.capacity_limit_low = capacity_limit_low
        self.capacity_limit_high = capacity_limit_high
        self.uncertainty_adjustment = uncertainty_adjustment

    def policy(self, state):
        wind_power = state[0]
        soc = np.array([state[1]])
        dac_capacity = state[2:2+self.num_units]
        dac_power_prev = state[2+self.num_units:]
        discharge_power = min((soc - self.soc_min) / (self.dt / 60), np.array([self.power_max]))
        power_available = (1 - self.uncertainty_adjustment) * wind_power + discharge_power
        operating_state = np.sign(dac_power_prev)
        operating_state = np.where(operating_state == 0, 1, operating_state)
        operating_state = np.where(dac_capacity / self.max_capacity < self.capacity_limit_low,
                                   1, operating_state)
        operating_state = np.where(dac_capacity / self.max_capacity > self.capacity_limit_high,
                                   -1, operating_state)
        dac_power = np.where(operating_state > 0, self.adsorption_power, self.desorption_power)
        deficit = dac_power.sum() - power_available
        unit = 0
        while deficit > 0.01:
            dac_power[unit] -= min(dac_power[unit], deficit)
            deficit = dac_power.sum() - power_available
            unit += 1
        dac_power = np.clip(operating_state * dac_power,
                            -self.desorption_power,
                            self.adsorption_power)
        if deficit < -0.01:
            battery_power = max(deficit,
                                (soc - self.soc_max) / (self.dt / 60),
                                -np.array([self.power_max]))
        else:
            battery_power = discharge_power
        if battery_power.ndim != dac_power.ndim:
            print("hi")
        return np.concatenate((battery_power, dac_power))
