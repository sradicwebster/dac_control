import numpy as np
from omegaconf import DictConfig


class BaseKinetics:
    def __init__(self,
                 dt: int,
                 q_CO2_eq: DictConfig,
                 q_H2O_eq: DictConfig,
                 ):
        self.dt = dt
        self.q_CO2_eq = q_CO2_eq
        self.q_H2O_eq = q_H2O_eq

    def step(self,
             component: str,
             q_i: np.ndarray,
             mode: np.ndarray,
             q_eq: dict,
             ):
        pass


class Linear(BaseKinetics):
    def __init__(self,
                 dt: int,
                 q_CO2_eq: DictConfig,
                 q_H2O_eq: DictConfig,
                 max_adsorption_rate: float,
                 ambient_adsorption_rate: float,
                 desorption_rate: float,
                 ):
        """

        Args:
            dt: time step (mins)
            q_CO2_eq: adsorption and desorption CO2 equilibrium loading per unit (kg)
            q_H2O_eq: adsorption and desorption H2O equilibrium loading per unit (kg)
            max_adsorption_rate: adsorption rate of CO2 with fan on per unit (kg / h)
            ambient_adsorption_rate: adsorption rate of CO2 with fan off per unit (kg / h)
            desorption_rate: desorption rate of CO2 per unit (kg / h)
        """
        super().__init__(dt, q_CO2_eq, q_H2O_eq)
        self.max_adsorption_rate = max_adsorption_rate
        self.ambient_adsorption_rate = ambient_adsorption_rate
        self.desorption_rate = desorption_rate

    def step(self,
             component: str,
             q_i: np.ndarray,
             mode: np.ndarray,
             q_eq: dict,
             ):
        # TODO add component dependency to adsorption/desorption rates
        adsorbed = np.select([mode < 0, mode == 0, mode > 0],
                             [-self.desorption_rate,
                              self.ambient_adsorption_rate,
                              self.max_adsorption_rate]) \
                   * self.dt / 60
        new_q = np.clip(q_i + adsorbed, q_eq["de"], q_eq["ad"])
        return new_q


class FirstOrder(BaseKinetics):
    def __init__(self,
                 dt: int,
                 q_CO2_eq: DictConfig,
                 q_H2O_eq: DictConfig,
                 k: DictConfig,
                 ):
        super().__init__(dt, q_CO2_eq, q_H2O_eq)
        self.k = k

    def _ode_sol(self, qi, qeq, k):
        return qeq - (qeq - qi) * np.exp(-k * self.dt * 60)

    def step(self,
             component: str,
             q_i: np.ndarray,
             mode: np.ndarray,
             q_eq: dict,
             ):
        new_q = np.select([mode < 0, mode == 0, mode > 0],
                          [self._ode_sol(q_i, q_eq["de"], self.k[component]["de"]),
                           q_i,
                           self._ode_sol(q_i, q_eq["ad"], self.k[component]["ad"])])
        return new_q
