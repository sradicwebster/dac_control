import os
import numpy as np
from typing import Optional


def load_wind_data(file: str,
                   dt: int,
                   var: Optional[float] = 0,
                   ) -> np.ndarray:
    """ Load wind data, perform interpolation and add Gaussian noise

    Args:
        file (str): file name within directory data/
        dt (int): time step (mins)
        var (float): variance of zero-mean Gaussian noise

    Returns:
        (np.ndarray): wind power series

    """
    wind_file = f"wind_power_dt{dt}_var{var}.npy"
    if wind_file in os.listdir("data"):
        wind_power_series = np.load(os.path.join("data", wind_file))
    else:
        power = np.load(f"data/{file}.npy")
        wind_power_series = power_interpolation(power, dt, var)
        np.save(os.path.join("data", wind_file), wind_power_series)
    return wind_power_series.reshape(-1, 1)


def power_interpolation(power: np.ndarray,
                        dt: int,
                        var: Optional[float] = 0,
                        ):
    """ If dt != 60 perform linear interpolation and add Gaussian noise

    Args:
        power (np.ndarray): wind power series
        dt (int): time step (mins)
        var (float): variance of zero-mean Gaussian noise

    Returns:
        (np.ndarray): wind power series at dt resolution

    """
    rng = np.random.RandomState(1)
    max_power = np.max(power)
    new_len = int((60 / dt) * (len(power) - 1) + 1)
    power_interp = np.interp(np.linspace(0, len(power) - 1, new_len),
                             np.linspace(0, len(power) - 1, len(power)),
                             power)
    if var != 0:
        noise = rng.normal(0, var, new_len)
        noise[np.arange(0, new_len, int(60 / dt))] = 0
        power_interp = np.clip(power_interp + noise, 0, max_power).round()
    return power_interp
