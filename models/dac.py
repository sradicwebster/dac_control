import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from typing import Optional

from .isotherm import equilibrium_loadings


class DAC:
    """
    Direct air capture model
    """
    def __init__(self,
                 num_units: int,
                 sorbent: str,
                 process_conditions: DictConfig,
                 dac_sizing_cfg: DictConfig,
                 kinetics_cfg: DictConfig,
                 ) -> None:
        """

        Args:
            num_units (int): number of DAC units
            sorbent (str): sorbent name
            process_conditions (DictConfig): adsorption and desorption process conditions including
                temperature and pressure
            dac_sizing_cfg (DictConfig): sizing parameters
            kinetics_cfg (DictConfig): kinetics parameters
        """
        self.num_units = num_units
        self.sorbent = sorbent
        self.process_conditions = process_conditions
        self.M_CO2 = 0.044009
        self.M_H2O = 0.018015
        q_CO2_eq, q_H2O_eq = equilibrium_loadings(sorbent, process_conditions)
        self.dac_sizing = instantiate(dac_sizing_cfg, q_CO2_eq=q_CO2_eq, _recursive_=False)
        assert self.dac_sizing.m_sorbent is not None,\
            "dac sizing class must populate mass of sorbent upon initialisation"
        self.q_CO2_eq = {mode: q_CO2_eq[mode] * self.M_CO2 * self.dac_sizing.m_sorbent
                         for mode in q_CO2_eq.keys()}
        self.q_H2O_eq = {mode: q_H2O_eq[mode] * self.M_H2O * self.dac_sizing.m_sorbent
                         for mode in q_H2O_eq.keys()}
        self.kinetics = instantiate(kinetics_cfg, q_CO2_eq=self.q_CO2_eq, q_H2O_eq=self.q_H2O_eq,
                                    m_sorbent=self.dac_sizing.m_sorbent)

        self.q_CO2 = None
        self.q_H2O = None
        self.T_units = None
        self.CO2_captured = None

    def reset(self,
              n: Optional[int] = 1,
              ) -> np.ndarray:
        """ Sets the sorbent loading to the minimum value for all DAC units

        Args:
            n: number of parallel experiments

        Returns:
            (np.ndarray): sorbent loading as a fraction of maximum loading in an array with shape
                n x number of DAC units

        """
        self.q_CO2 = np.ones((n, self.num_units)) * self.q_CO2_eq["de"]
        self.q_H2O = np.ones((n, self.num_units)) * self.q_H2O_eq["de"]
        self.T_units = np.ones((n, self.num_units)) * self.process_conditions.T_ad
        return self.q_CO2 / self.q_CO2_eq["ad"]

    def step(self,
             mode: np.ndarray,
             update_state: Optional[bool] = True,
             return_power: Optional[bool] = False,
             ) -> np.ndarray:
        """ Calculates the loading and power requirement of the DAC loading at the next time step
         according to the operation mode (-1 is desorption, 0 is ambient adsorption and +1 is fan
         powered adsorption)

        Args:
            mode (np.ndarray): mode to apply to the DAC units
            update_state (bool, optional): if 'True' update loading state
            return_power (bool, optional): if 'True' return the power requirement else return the
                sorbent loading

        Returns:
            (np.ndarray): loading or power requirement as an array with shape n x number of units

        """
        if mode.ndim == 1:
            mode = mode.reshape(1, -1)
        T_units_next, t_desorb = self.dac_sizing.temps_desorb_time(mode, self.T_units, self.q_CO2,
                                                                   self.q_H2O)
        q_CO2_next = self.kinetics.step(mode, "CO2", self.q_CO2, t_desorb)
        q_H2O_next = self.kinetics.step(mode, "H2O", self.q_H2O, t_desorb)
        power = self.dac_sizing.power_requirement(mode, self.q_CO2, self.q_H2O, q_CO2_next,
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
