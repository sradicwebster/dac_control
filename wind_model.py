import numpy as np


class BaseModel:
    def __init__(self,
                 wind_power: np.ndarray,
                 ):
        self.wind_power = wind_power

    def next(self,
             state: np.ndarray,
             ) -> np.ndarray:
        pass


class Constant(BaseModel):

    def next(self,
             state: np.ndarray,
             ) -> np.ndarray:
        return state[:, 0].reshape(-1, 1) * np.max(self.wind_power)


class Known(BaseModel):
    def __init__(self,
                 wind_power: np.ndarray,
                 horizon: int,
                 iterations: int
                 ):
        super().__init__(wind_power)
        self.horizon = horizon
        self.iterations = iterations
        self.step = 1
        self.i = 0
        self.h = 0

    def next(self,
             state: np.ndarray,
             ) -> np.ndarray:
        wind = np.repeat(self.wind_power[self.step + self.h].reshape(-1, 1), len(state), axis=0)
        self.h += 1
        if self.h == self.horizon:
            self.h = 0
            self.i += 1
        if self.i == self.iterations:
            self.i = self.h = 0
            self.step += 1
        return wind
