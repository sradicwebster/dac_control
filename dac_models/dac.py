import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from .isotherm import equilibrium_loadings


class DAC:
    def __init__(self,
                 num_units: int,
                 sorbent: str,
                 process_conditions: DictConfig,
                 unit_sizing_cfg: DictConfig,
                 kinetics_cfg: DictConfig,
                 ):
        self.num_units = num_units
        self.sorbent = sorbent
        self.process_conditions = process_conditions
        self.M_CO2 = 0.044009
        self.M_H2O = 0.018015
        q_CO2_eq, q_H2O_eq = equilibrium_loadings(sorbent, process_conditions)
        self.unit_sizing = instantiate(unit_sizing_cfg, q_CO2_eq=q_CO2_eq, _recursive_=False)
        assert self.unit_sizing.m_sorbent is not None,\
            "Unit sizing class must populate mass of sorbent upon initialisation"
        self.q_CO2_eq = {mode: q_CO2_eq[mode] * self.M_CO2 * self.unit_sizing.m_sorbent
                         for mode in q_CO2_eq.keys()}
        self.q_H2O_eq = {mode: q_H2O_eq[mode] * self.M_H2O * self.unit_sizing.m_sorbent
                         for mode in q_H2O_eq.keys()}
        self.kinetics = instantiate(kinetics_cfg, q_CO2_eq=self.q_CO2_eq, q_H2O_eq=self.q_H2O_eq,
                                    m_sorbent=self.unit_sizing.m_sorbent)

        self.q_CO2 = None
        self.q_H2O = None
        self.T_units = None
        self.CO2_captured = None

    def reset(self, n=1) -> np.ndarray:
        self.q_CO2 = np.ones((n, self.num_units)) * self.q_CO2_eq["de"]
        self.q_H2O = np.ones((n, self.num_units)) * self.q_H2O_eq["de"]
        self.T_units = np.ones((n, self.num_units)) * self.process_conditions.T_ad
        return self.q_CO2 / self.q_CO2_eq["ad"]

    def step(self,
             mode: np.ndarray,
             update_state: bool = True,
             return_power: bool = False,
             ) -> np.ndarray:
        if mode.ndim == 1:
            mode = mode.reshape(1, -1)
        T_units_next, t_desorb = self.unit_sizing.temps_desorb_time(mode, self.T_units, self.q_CO2,
                                                                    self.q_H2O)
        q_CO2_next = self.kinetics.step(mode, "CO2", self.q_CO2, t_desorb)
        q_H2O_next = self.kinetics.step(mode, "H2O", self.q_H2O, t_desorb)
        power = self.unit_sizing.power_requirement(mode, self.q_CO2, self.q_H2O, q_CO2_next,
                                                   q_H2O_next, self.T_units)
        if update_state:
            self.CO2_captured = np.maximum(self.q_CO2 - q_CO2_next, 0.0).sum(axis=1)
            self.q_CO2 = q_CO2_next
            self.q_H2O = q_H2O_next
            self.T_units = T_units_next
        if return_power:
            return power.sum(axis=1, keepdims=True)
        else:
            return q_CO2_next / self.q_CO2_eq["ad"]
