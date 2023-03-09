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
        if isinstance(self.process_conditions, tuple):
            self.process_conditions = self.process_conditions[0]
        self.M_CO2 = 0.044009
        self.M_H2O = 0.018015
        self.m_sorbent = None

    def temps_desorb_time(self,
                          mode: np.ndarray,
                          T: np.ndarray,
                          m_CO2: np.ndarray,
                          m_H2O: np.ndarray,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        T_next = np.where(mode == -1,
                          self.process_conditions["T_de"],
                          self.process_conditions["T_ad"])
        t_desorb = np.where(mode == -1, self.dt, 0)
        return T_next, t_desorb

    def power_requirement(self,
                          mode: np.ndarray,
                          m_CO2: np.ndarray,
                          m_H2O: np.ndarray,
                          m_CO2_next: np.ndarray,
                          m_H2O_next: np.ndarray,
                          T_units: np.ndarray,
                          T_units_next: np.ndarray,
                          ) -> np.ndarray:
        pass


class ConstantPower(BaseSizing):

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
            q_CO2_eq: adsorption and desorption CO2 equilibrium loading (mol_CO2/kg_sorbent)
            CO2_per_cycle: CO2 captured for an adsorption desorption cycle per unit (kg_CO2)
            fan_power: power requirement of air blower per unit (kW)
            desorption_power: power requirement of desorption per unit (kW)
        """
        super().__init__(dt, q_CO2_eq, process_conditions)
        self.CO2_per_cycle = CO2_per_cycle
        self.fan_power = fan_power
        self.desorption_power = desorption_power
        self.m_sorbent = self.CO2_per_cycle / (self.M_CO2 * (self.q_CO2_eq["ad"] -
                                                             self.q_CO2_eq["de"]))

    def power_requirement(self,
                          mode: np.ndarray,
                          m_CO2: np.ndarray,
                          m_H2O: np.ndarray,
                          m_CO2_next: np.ndarray,
                          m_H2O_next: np.ndarray,
                          T_units: np.ndarray,
                          T_units_next: np.ndarray,
                          ) -> np.ndarray:
        power = np.select([mode == -1, mode == 0, mode == 1],
                          [self.desorption_power, 0.0, self.fan_power])
        return power


class Detailed(BaseSizing):
    def __init__(self,
                 dt: int,
                 q_CO2_eq: DictConfig,
                 process_conditions: DictConfig,
                 sorbent: str,
                 P_heater: float,
                 dq_max_cfg: DictConfig,
                 geometry: DictConfig,
                 CO2_per_cycle: float = None,
                 ):
        super().__init__(dt, q_CO2_eq, process_conditions)
        self.P_heater = P_heater
        self.dq_max_cfg = dq_max_cfg
        self.geometry = geometry

        self.R = 8.314
        self.M_air = 0.02884
        self.mu_air = 1.789e-5
        self.p_ad = process_conditions.p_ad
        self.T_ad = process_conditions.T_ad
        self.p_de = process_conditions.p_de
        self.T_de = process_conditions.T_de
        self.CO2_conc = process_conditions.CO2_conc
        self.rho_air = self.p_ad * 1e5 * self.M_air / (self.R * (self.T_ad + 273))

        sorbent_props = pd.read_csv("data/sorbent_properties.csv").T
        sorbent_props = sorbent_props.rename(columns=sorbent_props.iloc[0]).drop("Solid_sorbent")
        assert sorbent in sorbent_props.index
        self.prop = sorbent_props.loc[sorbent].infer_objects()

        if CO2_per_cycle is not None:
            self.m_sorbent = self._sorbent_mass_from_rate(CO2_per_cycle)
            self.geometry.volume = self._volume_from_sorbent_mass(self.m_sorbent, geometry.void_frac)

        else:
            self.m_sorbent = self._sorbent_mass_from_geometry(geometry)
        self.Q = self._air_flow_rate(self.m_sorbent)

        # TODO sort out this mess
        deltaP_honey = self._pressure_drop_honeycomb(self.Q, geometry)
        deltaP_packed = self._pressure_drop_packed_bed(self.Q, 0.02)
        P_ad_honey = self._fan_power(deltaP_honey, self.Q)
        P_ad_packed = self._fan_power(deltaP_packed, self.Q)
        self.P_ad = P_ad_packed

    def _sorbent_mass_from_geometry(self,
                                    geometry: DictConfig,
                                    ) -> float:
        V_sorb = (1 - geometry.void_frac) * geometry.volume
        return self.prop["fb"] * self.prop["rho_p"] * V_sorb

    def _sorbent_mass_from_rate(self,
                                CO2_per_cycle: float,
                                ) -> float:
        return CO2_per_cycle / (self.M_CO2 * (self.q_CO2_eq["ad"] - self.q_CO2_eq["de"]))

    def _volume_from_sorbent_mass(self,
                                  m_sorbent: float,
                                  void_frac: float,
                                  ) -> float:
        V_sorb = m_sorbent / (self.prop["fb"] * self.prop["rho_p"])
        return float(V_sorb / (1 - void_frac))

    def _air_flow_rate(self,
                       m_sorbent: float,
                       ) -> float:
        dq_max = call(self.dq_max_cfg, q_CO2_eq=self.q_CO2_eq)
        Q_req = dq_max / (self.CO2_conc * 1e-6) * self.M_air / self.rho_air
        return Q_req * m_sorbent

    def temps_desorb_time(self,
                          mode: np.ndarray,
                          T: np.ndarray,
                          m_CO2: np.ndarray,
                          m_H2O: np.ndarray,
                          ) -> Tuple[np.ndarray, np.ndarray]:
        cp_sorb = self.prop["cp_s"] * 1e-3
        E_sen = self._sensible_heat(T, self.T_de, m_CO2, m_H2O)
        E_sorb = cp_sorb * self.m_sorbent * (self.T_de - T)
        P_sorb = E_sorb / (E_sen + 1e-10) * self.P_heater
        heat_temp = T + P_sorb * self.dt * 60 / (cp_sorb * self.m_sorbent)
        T_next = np.where(mode == -1, heat_temp, self.T_ad)
        T_next = np.minimum(T_next, self.T_de)
        time_de = self.dt - cp_sorb * self.m_sorbent * (T_next - T) / ((P_sorb + 1e-10) * 60)
        time_de = np.where(mode == -1, time_de, 0)
        return T_next, time_de

    def power_requirement(self,
                          mode: np.ndarray,
                          m_CO2: np.ndarray,
                          m_H2O: np.ndarray,
                          m_CO2_next: np.ndarray,
                          m_H2O_next: np.ndarray,
                          T_units: np.ndarray,
                          T_units_next: np.ndarray,
                          ) -> np.ndarray:
        E_de = self._heat_of_reaction(m_CO2 - m_CO2_next, m_H2O - m_H2O_next) + \
               self._sensible_heat(T_units, T_units_next, m_CO2, m_H2O)
        P_de = E_de / (60 * self.dt)
        return np.select([mode == -1, mode == 0, mode == 1], [P_de, 0.0, self.P_ad])

    def _pressure_drop_honeycomb(self, Q, geometry):
        packing_density = np.pi * 3 ** 0.5 / 6
        d_p = self.prop["dp"] * 1e-3
        V_filter = (1 - geometry.void_frac) * geometry.volume
        V_skel = np.pi * (geometry.ro ** 2 - geometry.ri ** 2) / packing_density * geometry.l
        V_sor = np.pi * (geometry.ri ** 2 - (geometry.ri - d_p) ** 2) * geometry.l
        ep = 1 - V_sor / (V_skel + V_sor)
        n_cells = int(V_filter / (V_skel + V_sor))
        A_flow = n_cells * np.pi * (geometry.ri - d_p) ** 2
        v_air = Q / A_flow
        Re = self.rho_air * v_air * d_p / self.mu_air
        a = 650
        m = 1.27
        n1 = -0.21
        n2 = 0.04
        f = a * Re ** -0.5 * (1 - ep) * ep ** m * (2 * geometry.ri / d_p) ** n1 \
            * (geometry.l / d_p) ** n2
        deltaP = f * geometry.l * self.rho_air * v_air ** 2 / d_p
        return deltaP

    def _pressure_drop_packed_bed(self, Q, l):
        d_p = self.prop["dp"] * 1e-3
        V_p = self.m_sorbent / self.prop["rho_p"]
        A_flow = V_p / l
        v_air = Q / A_flow
        Re = self.rho_air * v_air * d_p / self.mu_air
        if Re < 40:
            f = 805 / Re
        else:
            f = 38 / Re**0.15
        deltaP = f * l * self.rho_air * v_air ** 2 / d_p
        return deltaP

    def _fan_power(self, deltaP, Q, eff=0.6):
        return deltaP * Q / (eff * 1000)

    def _sensible_heat(self, T, T_next, m_CO2, m_H2O):
        cp_co2 = (0.819 + 0.918) / 2
        cp_water = 4.2
        cp_steam = 1.93
        T_bp = 80
        E_sorb = self.prop["cp_s"] * 1e-3 * self.m_sorbent * (T_next - T)
        E_co2 = cp_co2 * m_CO2 * (T_next - T)
        E_water = cp_water * m_H2O * np.maximum(T_bp - T, 0)
        E_steam = cp_steam * m_H2O * (T_next - np.maximum(T_bp, T))
        return E_sorb + E_co2 + E_water + E_steam

    def _heat_of_reaction(self, dm_CO2, dm_H2O):
        E_co2reac = self.prop["dH_co2"] * dm_CO2 / self.M_CO2
        E_h2oreac = self.prop["dH_h2o"] * dm_H2O / self.M_H2O
        return E_co2reac + E_h2oreac


def dq_max_first_order(q_CO2_eq, k):
    return k * (q_CO2_eq["ad"] - q_CO2_eq["de"])


def dq_max_linear(q_CO2_eq, max_adsorption_rate):
    return max_adsorption_rate / 3600
