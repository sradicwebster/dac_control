import numpy as np
import pandas as pd
from omegaconf import DictConfig


class BaseSizing:

    def __init__(self,
                 q_CO2_eq: DictConfig,
                 ) -> None:
        self.q_CO2_eq = q_CO2_eq
        self.M_CO2 = 0.044009
        self.sorbent_mass = None

    def power_requirement(self,
                          mode: np.ndarray,
                          ) -> np.ndarray:
        pass


class CO2Rate(BaseSizing):

    def __init__(self,
                 q_CO2_eq: DictConfig,
                 CO2_per_cycle: float,
                 fan_power: float,
                 desorption_power: float,
                 ) -> None:
        """

        Args:
            q_CO2_eq: adsorption and desorption CO2 equilibrium loading (mol_CO2 / kg_sorbent)
            CO2_per_cycle: CO2 captured for a whole adsorption desorption cycle per unit (kg_CO2)
            fan_power: power requirement of air blower per unit (kW)
            desorption_power: power requirement of desorption per unit (kW)
        """
        super().__init__(q_CO2_eq)
        self.CO2_per_cycle = CO2_per_cycle
        self.fan_power = fan_power
        self.desorption_power = desorption_power
        self.sorbent_mass = self.CO2_per_cycle / (self.M_CO2 * (self.q_CO2_eq["ad"] -
                                                                self.q_CO2_eq["de"]))

    def power_requirement(self,
                          mode: np.ndarray,
                          ) -> np.ndarray:
        power = np.select([mode < 0, mode == 0, mode > 0],
                          [self.desorption_power, 0.0, self.fan_power])
        return power

