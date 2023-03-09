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
                 ) -> None:
        """

        Args:
            dt (int): time step (min)
            q_CO2_eq (DictConfig): CO2 equilibrium loadings
            q_H2O_eq (DictConfig): H2O equilibrium loadings
        """
        self.dt = dt
        self.q_eq = {"CO2": q_CO2_eq, "H2O": q_H2O_eq}

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
                 rates: DictConfig,
                 ) -> None:
        """ Linear kinetics

        Args:
            dt (int): time step (min)
            q_CO2_eq (DictConfig): adsorption and desorption CO2 equilibrium loading per unit
                (mol_CO2/kg_sorbent)
            q_H2O_eq (DictConfig): adsorption and desorption H2O equilibrium loading per unit
                (mol_H2O/kg_sorbent)
            rates (DictConfig): adsorption and desorption rates for CO2 and H2O (mol/kg_sorbent/h)
        """
        super().__init__(dt, q_CO2_eq, q_H2O_eq)
        self.rates = rates

    def step(self,
             mode: np.ndarray,
             comp: str,
             q_i: np.ndarray,
             t_desorb: np.ndarray,
             ) -> np.ndarray:
        """ Updates the DAC loading according to the adsorption/desorption kinetics

        Args:
            mode (np.ndarray): DAC unit mode of operation
            comp (str): component, either 'CO2' or 'H2O'
            q_i (np.ndarray): initial loading as an array with shape number of experiment x number
                of DAC units
            t_desorb (np.ndarray): time to desorb

        Returns:
            (np.ndarray): new loading as an array with shape number of experiment x number
                of DAC units

        """
        adsorb_rate = np.select([mode == -1, mode == 0, mode == 1],
                                [-self.rates[comp]["de"],
                                 self.rates[comp]["ad_amb"],
                                 self.rates[comp]["ad_max"]])
        adsorbed = adsorb_rate * np.where(mode == -1, t_desorb, self.dt) / 60
        new_q = np.clip(q_i + adsorbed, self.q_eq[comp]["de"], self.q_eq[comp]["ad"])
        return new_q


class FirstOrder(BaseKinetics):
    def __init__(self,
                 dt: int,
                 q_CO2_eq: DictConfig,
                 q_H2O_eq: DictConfig,
                 k: DictConfig,
                 ):
        """ First order kinetics as described by the linear driving force (LDF) model

        Args:
            dt (int): time step (mins)
            q_CO2_eq (DictConfig): equilibrium loading (mol_CO2/kg_sorbent)
            q_H2O_eq (DictConfig): equilibrium loading (mol_H2O/kg_sorbent)
            k (DictConfig): rate constants (1/s)
        """
        super().__init__(dt, q_CO2_eq, q_H2O_eq)
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
        new_q = np.select([mode == -1, mode == 0, mode == 1],
                          [self._ode_sol(q_i, self.q_eq[comp]["de"], self.k[comp]["de"], time),
                           self._ode_sol(q_i, self.q_eq[comp]["ad"], self.k[comp]["ad_amb"], time),
                           self._ode_sol(q_i, self.q_eq[comp]["ad"], self.k[comp]["ad_max"], time)])
        return new_q
