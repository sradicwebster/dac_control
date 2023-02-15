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
        self.unit_sizing = instantiate(unit_sizing_cfg, q_CO2_eq=q_CO2_eq)
        self.q_CO2_eq = {k: q_CO2_eq[k] * self.M_CO2 * self.unit_sizing.sorbent_mass
                         for k in q_CO2_eq.keys()}
        self.q_H2O_eq = {k: q_H2O_eq[k] * self.M_H2O * self.unit_sizing.sorbent_mass
                         for k in q_H2O_eq.keys()}
        self.kinetics = instantiate(kinetics_cfg, q_CO2_eq=self.q_CO2_eq, q_H2O_eq=self.q_H2O_eq)

        self.q_CO2 = None
        self.q_H2O = None
        self.CO2_captured = None

    def reset(self):
        self.q_CO2 = np.ones((1, self.num_units)) * self.q_CO2_eq["de"]
        self.q_H2O = np.ones((1, self.num_units)) * self.q_H2O_eq["de"]
        return self.q_CO2 / self.q_CO2_eq["ad"]

    def step(self,
             mode: np.ndarray,
             ):
        if mode.ndim == 1:
            mode = mode.reshape(1, -1)
        next_q_CO2 = self.kinetics.step("CO2", self.q_CO2, mode, self.q_CO2_eq)
        next_q_H20 = self.kinetics.step("H2O", self.q_H2O, mode, self.q_H2O_eq)
        self.CO2_captured = np.maximum(self.q_CO2 - next_q_CO2, 0.0).sum(axis=1)
        self.q_CO2 = next_q_CO2
        self.q_H2O = next_q_H20
        return next_q_CO2 / self.q_CO2_eq["ad"]

    def power_requirement(self, mode: np.ndarray):
        if mode.ndim == 1:
            mode = mode.reshape(1, -1)
        power = self.unit_sizing.power_requirement(mode)
        return power.sum(axis=1, keepdims=True)
