import numpy as np
from omegaconf import DictConfig


class BaseKinetics:
    """
    CO2 and H2O kinetics base class
    """
    def __init__(self,
                 dt: int,
                 q_CO2_eq: DictConfig,
                 q_H2O_eq: DictConfig,
                 m_sorbent: float,
                 ) -> None:
        self.dt = dt
        self.q_eq = {"CO2": q_CO2_eq, "H2O": q_H2O_eq}
        self.m_sorbent = m_sorbent

    def step(self,
             mode: np.ndarray,
             comp: str,
             q_i: np.ndarray,
             t_desorb: np.ndarray,
             ) -> np.ndarray:
        """ Updates the DAC loading according to the adsorption/desorption kinetics

        Args:
            mode (np.ndarray): DAC unit mode of operation
            comp (str): componet, either 'CO2' or 'H2O'
            q_i (np.ndarray): initial loading as an array with shape number of experiment x number
                of DAC units
            t_desorb (np.ndarray): time to desorb

        Returns:
            (np.ndarray): new loading as an array with shape number of experiment x number
                of DAC units

        """
        pass


class Linear(BaseKinetics):
    def __init__(self,
                 dt: int,
                 q_CO2_eq: DictConfig,
                 q_H2O_eq: DictConfig,
                 m_sorbent: float,
                 max_adsorption_rate: float,
                 ambient_adsorption_rate: float,
                 desorption_rate: float,
                 ) -> None:
        """ Linear kinetics

        Args:
            dt (int): time step (mins)
            q_CO2_eq (DictConfig): adsorption and desorption CO2 equilibrium loading per unit (kg)
            q_H2O_eq (DictConfig): adsorption and desorption H2O equilibrium loading per unit (kg)
            max_adsorption_rate (float): adsorption rate of CO2 with fan on (mol_CO2/kg_sorbent/h)
            ambient_adsorption_rate (float): adsorption rate of CO2 with fan off
                (mol_CO2/kg_sorbent/h)
            desorption_rate (float): desorption rate of CO2 (mol_CO2/kg_sorbent/h)
        """
        super().__init__(dt, q_CO2_eq, q_H2O_eq, m_sorbent)
        self.max_ad_rate = max_adsorption_rate
        self.amb_ad_rate = ambient_adsorption_rate
        self.de_rate = desorption_rate

    def step(self,
             mode: np.ndarray,
             comp: str,
             q_i: np.ndarray,
             t_desorb: np.ndarray,
             ) -> np.ndarray:
        """ Updates the DAC loading according to the adsorption/desorption kinetics

        Args:
            mode (np.ndarray): DAC unit mode of operation
            comp (str): componet, either 'CO2' or 'H2O'
            q_i (np.ndarray): initial loading as an array with shape number of experiment x number
                of DAC units
            t_desorb (np.ndarray): time to desorb

        Returns:
            (np.ndarray): new loading as an array with shape number of experiment x number
                of DAC units

        """
        M_CO2 = 0.044009
        adsorb_rate = np.select([mode < 0, mode == 0, mode > 0],
                                [-self.de_rate, self.amb_ad_rate, self.max_ad_rate]) \
                      * M_CO2 * self.m_sorbent
        adsorbed = adsorb_rate * np.where(mode == -1, t_desorb, self.dt) / 60
        new_q = np.clip(q_i + adsorbed, self.q_eq[comp]["de"], self.q_eq[comp]["ad"])
        return new_q


class FirstOrder(BaseKinetics):
    def __init__(self,
                 dt: int,
                 q_CO2_eq: DictConfig,
                 q_H2O_eq: DictConfig,
                 m_sorbent: float,
                 k: DictConfig,
                 ):
        """ First order kinetics as described by the linear driving force (LDF) model

        Args:
            dt (int): time step (mins)
            q_CO2_eq (DictConfig): equilibrium loading (mol_CO2/kg_sorbent)
            q_H2O_eq (DictConfig): equilibrium loading (mol_H2O/kg_sorbent)
            m_sorbent (float): mass of sorbent (kg)
            k (DictConfig): rate constants (1/s)
        """
        super().__init__(dt, q_CO2_eq, q_H2O_eq, m_sorbent)
        self.k = k

    def _ode_sol(self,
                 qi: np.ndarray,
                 qeq: np.ndarray,
                 k: float,
                 t: int,
                 ) -> np.ndarray:
        return qeq - (qeq - qi) * np.exp(-k * t)

    def step(self,
             mode: np.ndarray,
             comp: str,
             q_i: np.ndarray,
             t_desorb: np.ndarray,
             ) -> np.ndarray:
        """ Updates the DAC loading according to the adsorption/desorption kinetics

        Args:
            mode (np.ndarray): DAC unit mode of operation
            comp (str): componet, either 'CO2' or 'H2O'
            q_i (np.ndarray): initial loading as an array with shape number of experiment x number
                of DAC units
            t_desorb (np.ndarray): time to desorb

        Returns:
            (np.ndarray): new loading as an array with shape number of experiment x number
                of DAC units

        """
        time = np.where(mode == -1, t_desorb, self.dt) * 60
        new_q = np.select([mode < 0, mode == 0, mode > 0],
                          [self._ode_sol(q_i, self.q_eq[comp]["de"], self.k[comp]["de"], time),
                           self._ode_sol(q_i, self.q_eq[comp]["ad"], self.k[comp]["ad_amb"], time),
                           self._ode_sol(q_i, self.q_eq[comp]["ad"], self.k[comp]["ad_max"], time)])
        return new_q
