import os
import numpy as np
import pandas as pd
import argparse


def load_wind_data(dt):
    wind_file = f"wind_power_dt{dt}_var0.npy"
    if wind_file in os.listdir("data"):
        wind_power_series = np.load(os.path.join("data", wind_file))
    else:
        power = np.load("data/wind_power_1year.npy")
        wind_power_series = power_interpolation(power, dt)
        np.save(os.path.join("../data", wind_file), wind_power_series)
    return wind_power_series.reshape(-1, 1)


def power_interpolation(power: np.ndarray, dt: int, var: float = 0):
    rng = np.random.RandomState(1)
    max_power = power.max()
    new_len = int((60 / dt) * (len(power) - 1) + 1)
    power_interp = np.interp(np.linspace(0, len(power) - 1, new_len),
                             np.linspace(0, len(power) - 1, len(power)),
                             power)
    if var != 0:
        noise = rng.normal(0, var, new_len)
        noise[np.arange(0, new_len, int(60 / dt))] = 0
        power_interp = np.clip(power_interp + noise, 0, max_power).round()
    return power_interp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dt", type=int)
    parser.add_argument("--noise_var", type=float, default=500)
    args = parser.parse_args()
    power = pd.read_csv("data/cascadia_power_output.csv")["power_kw"].to_numpy()
    power_interp = power_interpolation(power, args.dt, args.noise_var)
    np.save(f"data/wind_power_dt{args.dt}_var{args.noise_var}", power_interp)
