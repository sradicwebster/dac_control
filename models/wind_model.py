import numpy as np


class BaseModel:
    """
    Base wind model
    """
    def __init__(self,
                 wind_power: np.ndarray,
                 ) -> None:
        """

        Args:
            wind_power (np.ndarray): wind power series (kW)
        """
        self.wind_power = wind_power

    def next(self,
             state: np.ndarray,
             ) -> np.ndarray:
        """ Predict the wind power for the next time step based on the current state

        Args:
            state (np.ndarray):

        Returns:
            (np.ndarray): wind power prediction

        """
        pass


class Constant(BaseModel):
    """
    Prediction constant wind
    """

    def next(self,
             state: np.ndarray,
             ) -> np.ndarray:
        """ Predict the wind power for the next time step based on the current state

        Args:
            state (np.ndarray):

        Returns:
            (np.ndarray): wind power prediction

        """
        return state[:, 0].reshape(-1, 1) * np.max(self.wind_power)


class Known(BaseModel):
    """
    Provide the known wind power
    """
    def __init__(self,
                 wind_power: np.ndarray,
                 horizon: int,
                 iterations: int
                 ) -> None:
        """

        Args:
            wind_power (np.ndarray): wind power series
            horizon (int): optimisation horizon for MPC controller
            iterations (int): planning iterations for population based optimiser
        """
        super().__init__(wind_power)
        self.horizon = horizon
        self.iterations = iterations
        self.step = 1
        self.i = 0
        self.h = 0

    def next(self,
             state: np.ndarray,
             ) -> np.ndarray:
        """ Predict the wind power for the next time step based on the current state

        Args:
            state (np.ndarray):

        Returns:
            (np.ndarray): wind power prediction

        """
        wind = np.repeat(self.wind_power[self.step + self.h].reshape(-1, 1), len(state), axis=0)
        self.h += 1
        if self.h == self.horizon:
            self.h = 0
            self.i += 1
        if self.i == self.iterations:
            self.i = self.h = 0
            self.step += 1
        return wind
