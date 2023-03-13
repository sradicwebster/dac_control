import numpy as np
from omegaconf import DictConfig
import stable_baselines3


class BaseController:
    """
    Base class for DAC system control
    """
    def policy(self,
               state: np.ndarray,
               ) -> np.ndarray:
        """ Provide the control action (DAC mode) based on the current state of the system

        Args:
            state (np.ndarray): current state

        Returns:
            (np.ndarray): control mode

        """
        pass
    
    
class LoadingRule(BaseController):
    """
    Rule based controller based on high and low loading limits
    """
    def __init__(self,
                 num_units: int,
                 loading_low: float,
                 loading_high: float,
                 ) -> None:
        """

        Args:
            num_units (int): number of DAC units
            loading_low (float): loading fraction to start adsorption
            loading_high (float): loading fraction to start desorption
        """
        self.num_units = num_units
        self.loading_low = loading_low
        self.loading_high = loading_high
        self.prev_mode = np.zeros(self.num_units)

    def policy(self,
               state: np.ndarray,
               ) -> np.ndarray:
        """ Provide the control action (DAC mode) based on the current state of the system

        Args:
            state (np.ndarray): current state

        Returns:
            (np.ndarray): control mode

        """
        loading = state[2:2+self.num_units]
        mode = np.where(self.prev_mode == 0, 1, self.prev_mode)
        mode = np.where(loading <= self.loading_low, 1, mode)
        mode = np.where(loading >= self.loading_high, -1, mode)
        self.prev_mode = mode
        return mode


class UnitCyclingRule(BaseController):
    """
    Cycle around the DAC so 1 is desorbing whilst the others are adsorbing
    """
    def __init__(self,
                 num_units: int,
                 dt: int,
                 regen_time: int,
                 max_steps: int,
                 ):
        """

        Args:
            num_units (int): number of DAC units
            dt (int): time step (mins)
            regen_time (int): desorption time (mins)
            max_steps (int): maximum number of steps a particular unit waits inbetween desorption,
                only affect if num_units > max_steps for example if num_units = 10 and max_steps =
                6 the controller iterates around dac number: [0, 1], [2, 3], [4, 5], [6, 7], [8],
                [9]

        """
        self.num_units = num_units
        self.unit_ops = np.array_split(np.arange(num_units), min(num_units, max_steps))
        self.regen_steps = np.ceil(regen_time / dt).astype(int)
        self.unit_op = 0
        self.step = 0

    def policy(self,
               state: np.ndarray,
               ) -> np.ndarray:
        """ Provide the control action (DAC mode) based on the current state of the system

        Args:
            state (np.ndarray): current state

        Returns:
            (np.ndarray): control mode

        """
        controls = np.ones(self.num_units)
        controls[self.unit_ops[self.unit_op]] *= -1
        self.step += 1
        if self.step == self.regen_steps:
            self.unit_op = (self.unit_op + 1) % len(self.unit_ops)
            self.step = 0
        return controls


class DiscreteCEM(BaseController):
    """
    Discrete cross entropy method stochastic optimisation
    """
    def __init__(self,
                 model,
                 horizon: int,
                 population_size: int,
                 elite_frac: float,
                 iterations: int,
                 alpha: float,
                 replan: bool,
                 desorb_pen: float,
                 ):
        """

        Args:
            model: dynamics model instance
            horizon (int): number of time steps in the prediction/control horizon
            population_size (int): population size
            elite_frac (float): fraction of the population to use for action distrubution update,
                elite population size = int(elite_frac * population_size)
            iterations:
            alpha (float): Polyak average weighting parameter,
                param_new = alpha * param_old + (1 - alpha) * param_estimate
            replan (bool): if 'True' replan every timestep from a uniform distribution, if 'False'
                keep the action distribution from the previous time step
            desorb_pen (float): reward function penalty for starting to desorb
        """
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
               ) -> np.ndarray:
        """ Provide the control action (DAC mode) based on the current state of the system

        Args:
            state (np.ndarray): current state

        Returns:
            (np.ndarray): control mode

        """
        self.model.dac.reset(n=self.population_size)
        self.model.battery.reset(n=self.population_size)

        if self.replan:
            self.action_prob = np.ones((self.horizon, self.num_units, 3)) / 3
        else:
            self.action_prob = np.concatenate((self.action_prob[1:, ...],
                                               np.ones((1, self.num_units, 3)) / 3))

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
                rewards[:, h] = self.model.dac.CO2_captured * 1e-3 - self.desorb_pen * start_desorb
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


class RLAgent(BaseController):
    """
    Deploy reinforcement learning agent trained using Stable Baselines 3
    """
    def __init__(self,
                 algorithm: DictConfig,
                 training_timesteps: int,
                 wandb_name: str,
                 ) -> None:
        """

        Args:
            algorithm (DictConfig): algorithm parameters
            training_timesteps (int): number of training time steps
            wandb_name (str): Weights & Biases name for training
        """
        self.agent = eval(algorithm._target_).load(f"trained_agents/{wandb_name}")

    def policy(self,
               state: np.ndarray,
               ) -> np.ndarray:
        """ Provide the control action (DAC mode) based on the current state of the system

        Args:
            state (np.ndarray): current state

        Returns:
            (np.ndarray): control mode

        """
        return self.agent.predict(state, deterministic=True)[0] - 1
