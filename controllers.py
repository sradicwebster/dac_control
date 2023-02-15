import numpy as np
import omegaconf

from dynamics_model import BaseDynamics


class BaseController:
    def __init__(self,
                 model: BaseDynamics):
        self.model = model
        
    def policy(self,
               state: np.ndarray):
        pass
    
    
class SorbentLoadingRule(BaseController):
    def __init__(self,
                 model: BaseDynamics,
                 num_units: int,
                 loading_low: float,
                 loading_high: float,
                 ):
        super().__init__(model)
        """

        Args:
            num_units: number of DAC units
            loading_low: loading fraction to start adsorption
            loading_high: loading fraction to start desorption
        """
        self.num_units = num_units
        self.loading_low = loading_low
        self.loading_high = loading_high
        self.prev_mode = np.zeros(self.num_units)

    def policy(self,
               state: np.ndarray):
        """

        Args:
            state:

        Returns:

        """
        loading = state[2:2+self.num_units]
        mode = np.where(self.prev_mode == 0, 1, self.prev_mode)
        mode = np.where(loading <= self.loading_low, 1, mode)
        mode = np.where(loading >= self.loading_high, -1, mode)
        self.prev_mode = mode
        return mode


class DiscreteCrossEntropyMethod:

    def __init__(self,
                 model: BaseDynamics,
                 horizon: int,
                 population_size: int,
                 elite_frac: float,
                 iterations: int,
                 alpha: float,
                 replan: bool,
                 desorb_pen: float,
                 ):
        self.model = model
        self.horizon = horizon
        self.population_size = population_size
        self.elite_num = int(elite_frac * population_size)
        self.iterations = iterations
        self.alpha = alpha
        self.replan = replan
        self.desorb_pen = desorb_pen
        self.num_units = model.num_units
        self.action_prob = np.ones((self.horizon, self.num_units, 3)) / 3

    def policy(self,
               state: np.ndarray,
               ):
        self.model.battery.soc = np.repeat(state[1].reshape(1, -1), self.population_size, axis=0)
        self.model.dac.loading = np.repeat(state[2: 2 + self.model.dac.num_units].reshape(1, -1),
                                           self.population_size, axis=0)
        if self.replan:
            self.action_prob = np.ones((self.horizon, self.num_units, 3)) / 3
        else:
            self.action_prob = np.concatenate((self.action_prob[1:, ...],
                                               np.ones((1, self.num_units, 3)) / 3))

        prob_add = 0
        for u in range(self.num_units):
            self.action_prob[0][u, (state[-self.num_units:].astype(int) + 1)[u]] += prob_add
        self.action_prob[0] /= (1 + prob_add)

        for i in range(self.iterations):
            state_pop = np.repeat(state.reshape(1, -1), self.population_size, axis=0)
            rewards = np.zeros((self.population_size, self.horizon))
            population = np.zeros((self.population_size, self.horizon, self.num_units), dtype=int)
            for h in range(self.horizon):
                controls = np.stack([np.random.choice(np.arange(-1, 2),
                                                      self.population_size,
                                                      p=self.action_prob[h, u])
                                     for u in range(self.num_units)]).T
                population[:, h, :] = controls
                next_state_pop = self.model.step(state_pop, controls)
                start_desorb = np.logical_and(state_pop[:, -self.num_units:] != -1,
                                              controls == -1).sum(1)
                rewards[:, h] = self.model.dac.captured - self.desorb_pen * start_desorb
                state_pop = next_state_pop
            elite = population[np.flip(np.argsort(rewards.sum(axis=1)))[:self.elite_num]]
            for h in range(self.horizon):
                for u in range(self.num_units):
                    counts = np.unique(np.concatenate((elite[:, h, u], np.array([-1, 0, 1]))),
                                       return_counts=True)[1]
                    probs = counts / counts.sum()
                    self.action_prob[h, u] = self.alpha * self.action_prob[h, u] +\
                                             (1 - self.alpha) * probs
        return np.argmax(self.action_prob[0], axis=1) - 1
