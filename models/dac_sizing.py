import numpy as np
import pandas as pd
from omegaconf import DictConfig
from hydra.utils import call
from typing import Tuple


class BaseSizing:

    def __init__(self,
                 dt: int,
                 q_CO2_eq: DictConfig,
                 process_conditions: DictConfig,
                 ) -> None:
        self.dt = dt
        self.q_CO2_eq = q_CO2_eq
        self.process_conditions = process_conditions,
        # for some reason process_conditions is being turned into a tuple
        if isinstance(self.process_conditions, tuple):
            self.process_conditions = self.process_conditions[0]
        self.M_CO2 = 0.044009
        self.M_H2O = 0.018015
        self.m_sorbent = None

    def temps_desorb_time(self,
                          mode: np.ndarray,
                          T: np.ndarray,
                          q_CO2: np.ndarray,
                          q_H2O: np.ndarray,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        T_next = np.where(mode == -1, self.process_conditions["T_de"],
                          self.process_conditions["T_ad"])
        t_desorb = np.where(mode == -1, self.dt, 0)
        return T_next, t_desorb

    def power_requirement(self,
                          mode: np.ndarray,
                          q_CO2: np.ndarray,
                          q_H2O: np.ndarray,
                          q_CO2_next: np.ndarray,
                          q_H2O_next: np.ndarray,
                          T_units: np.ndarray,
                          ) -> np.ndarray:
        pass


class CO2Rate(BaseSizing):

    def __init__(self,
                 dt: int,
                 q_CO2_eq: DictConfig,
                 process_conditions: DictConfig,
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
        super().__init__(dt, q_CO2_eq, process_conditions)
        self.CO2_per_cycle = CO2_per_cycle
        self.fan_power = fan_power
        # TODO change to desorption power / kg CO2??
        self.desorption_power = desorption_power
        self.m_sorbent = self.CO2_per_cycle / (self.M_CO2 * (self.q_CO2_eq["ad"] -
                                                             self.q_CO2_eq["de"]))

    def power_requirement(self,
                          mode: np.ndarray,
                          q_CO2: np.ndarray,
                          q_H2O: np.ndarray,
                          q_CO2_next: np.ndarray,
                          q_H2O_next: np.ndarray,
                          T_units: np.ndarray,
                          ) -> np.ndarray:
        power = np.select([mode < 0, mode == 0, mode > 0],
                          [self.desorption_power, 0.0, self.fan_power])
        return power


class Detailed(BaseSizing):
    def __init__(self,
                 dt: int,
                 q_CO2_eq: DictConfig,
                 process_conditions: DictConfig,
                 v_air: float,
                 P_heater: float,
                 geometry: DictConfig,
                 dq_max_cfg: DictConfig,
                 sorbent: str,

                 ):
        super().__init__(dt, q_CO2_eq, process_conditions)
        self.v_air = v_air
        self.P_heater = P_heater
        self.dq_max_cfg = dq_max_cfg

        self.R = 8.314
        self.M_air = 0.02884
        self.p_ad = process_conditions.p_ad
        self.T_ad = process_conditions.T_ad
        self.p_de = process_conditions.p_de
        self.T_de = process_conditions.T_de
        self.CO2_conc = process_conditions.CO2_conc
        self.rho_air = self.p_ad * 1e5 * self.M_air / (self.R * (self.T_ad + 273))
        self.mu_air = 1.789e-5
        sorbent_props = pd.read_csv("data/sorbent_properties.csv").T
        sorbent_props = sorbent_props.rename(columns=sorbent_props.iloc[0]).drop("Solid_sorbent")\
            .infer_objects()
        assert sorbent in sorbent_props.index
        prop = sorbent_props.loc[sorbent]
        self.d_sorb = float(prop["dp"]) * 1e-3

        Q = self._air_flow_rate(v_air, geometry)
        self.m_sorbent = self._sorbent_mass(Q)
        self.P_ad = self._fan_power(Q, geometry)

    def _sorbent_mass(self, Q: float) -> float:
        dq_max = call(self.dq_max_cfg, q_CO2_eq=self.q_CO2_eq)
        Q_req = dq_max / (self.CO2_conc * 1e-6) * self.M_air / self.rho_air
        return Q / Q_req

    def temps_desorb_time(self,
                          mode: np.ndarray,
                          T: np.ndarray,
                          q_CO2: np.ndarray,
                          q_H2O: np.ndarray,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        cp_al = 0.91
        E_skel = cp_al * self.m_skel * (self.T_de - T)
        E_sen = self._sensible_heat(T, q_CO2, q_H2O)
        P_skel = E_skel / (E_sen + 1e-10) * self.P_heater
        T_next = np.where(mode == -1, T + P_skel * self.dt * 60 / (cp_al * self.m_skel), self.T_ad)
        t_de = np.where(mode == -1,
                        self.dt - cp_al * self.m_skel * (T_next - self.T_de) / ((P_skel + 1e-10) * 60),
                        0)
        T_next = np.minimum(T_next, self.T_de)
        return T_next, t_de

    def power_requirement(self,
                          mode: np.ndarray,
                          q_CO2: np.ndarray,
                          q_H2O: np.ndarray,
                          q_CO2_next: np.ndarray,
                          q_H2O_next: np.ndarray,
                          T_units: np.ndarray,
                          ) -> np.ndarray:
        E_de = self._heat_of_reaction(q_CO2 - q_CO2_next, q_H2O - q_H2O_next) + \
               self._sensible_heat(T_units, q_CO2, q_H2O)
        # TODO what to do when P_de > P_heater?
        #P_de = np.minimum(E_de / (60 * self.dt), self.P_heater)
        P_de = E_de / (60 * self.dt)
        return np.select([mode < 0, mode == 0, mode > 0], [P_de, 0.0, self.P_ad])

    def _air_flow_rate(self, v_air: float, geometry: DictConfig) -> float:
        rho_al = 2700
        packing_density = np.pi * 3**0.5 / 6
        V_filter = (1 - geometry.void_frac) * geometry.unit_volume
        V_skel = np.pi * (geometry.ro**2 - geometry.ri**2) / packing_density * geometry.l
        V_sor = np.pi * (geometry.ri**2 - (geometry.ri - self.d_sorb)**2) * geometry.l
        self.ep = 1 - V_sor / (V_skel + V_sor)
        n_cells = int(V_filter / (V_skel + V_sor))
        self.m_skel = rho_al * V_skel * n_cells
        A_flow = n_cells * np.pi * (geometry.ri - self.d_sorb) ** 2
        return v_air * A_flow

    def _fan_power(self,
                   Q: float,
                   geometry: DictConfig,
                   ):
        Re = self.rho_air * self.v_air * self.d_sorb / self.mu_air
        a = 650
        m = 1.27
        n1 = -0.21
        n2 = 0.04
        f = a * Re ** -0.5 * (1 - self.ep) * self.ep ** m * (2 * geometry.ri / self.d_sorb) ** n1 \
            * (geometry.l / self.d_sorb) ** n2
        deltaP = f * geometry.l * self.rho_air * self.v_air ** 2 / self.d_sorb
        fan_eff = 0.6
        return deltaP * Q / (fan_eff * 1000)

    def _sensible_heat(self, T, q_CO2, q_H2O):
        cp_sorbent = 1.58
        cp_co2 = (0.819 + 0.918) / 2
        cp_water = 4.2
        cp_steam = 1.93
        cp_al = 0.91
        T_bp = 80
        E_sorb = cp_sorbent * self.m_sorbent * (self.T_de - T)
        E_skel = cp_al * self.m_skel * (self.T_de - T)
        E_co2 = cp_co2 * q_CO2 * (self.T_de - T)
        E_water = cp_water * q_H2O * np.maximum(T_bp - T, 0)
        E_steam = cp_steam * q_H2O * (self.T_de - np.maximum(T_bp, T))
        return E_sorb + E_skel + E_co2 + E_water + E_steam

    def _heat_of_reaction(self, dq_CO2, dq_H2O):
        dH_co2 = 58.2
        dH_h2o = 53.2
        E_co2reac = dH_co2 * dq_CO2 / self.M_CO2
        E_h2oreac = dH_h2o * dq_H2O / self.M_H2O
        return E_co2reac + E_h2oreac


def dq_max_first_order(q_CO2_eq, k):
    return k * (q_CO2_eq["ad"] - q_CO2_eq["de"])


def dq_max_linear(q_CO2_eq, max_adsorption_rate):
    return max_adsorption_rate / 3600
