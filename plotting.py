from matplotlib import pyplot as plt
import numpy as np


def plot_results(all_states,
                 all_actions,
                 all_rewards,
                 num_units,
                 hours,
                 dt,
                 dac_loading_max,
                 battery_capacity,
                 ):

    iters = int(hours * 60 / dt)
    fig, ax = plt.subplots(num_units + 3, figsize=(12, 8))
    ax[0].set_title("Wind power")
    ax[0].plot(np.linspace(0, hours, iters + 1), all_states[:, 0], label='Wind', c='g')
    ax[0].set_xlim(0, hours)
    ax[0].set_ylim(0, np.max(all_states[:, 0]) + 1000)
    ax[0].set_ylabel("Power (kW)")
    ax[1].set_title("Battery")
    ax[1].plot(np.linspace(0, hours, iters + 1), all_states[:, 1] / battery_capacity, c='r')
    ax[1].set_xlim(0, hours)
    ax[1].set_ylim(0, 1.1)
    ax[1].set_ylabel("SOC")
    for i in range(num_units):
        ax[i + 2].set_title(f"DAC #{i + 1}")
        ax[i + 2].plot(np.linspace(0, hours, iters, endpoint=False), all_actions[:, i],
                       label='DAC mode', c='b')
        ax[i + 2].set_xlim(0, hours)
        ax[i + 2].set_ylim(-1.1, 1.1)
        ax[i + 2].set_ylabel("Mode")
        ax[i + 2].legend(loc='upper left')
        ax2r = ax[i + 2].twinx()
        ax2r.plot(np.linspace(0, hours, iters + 1), all_states[:, i + 2] / dac_loading_max,
                  label='Capacity', c='r')
        ax2r.set_ylim(0, 1.1)
        ax2r.set_ylabel("Capacity")
        ax2r.legend(loc='upper right')
    ax[num_units + 2].set_title("CO2 captured")
    ax[num_units + 2].plot(np.linspace(0, hours, iters, endpoint=False), all_rewards.flatten())
    ax[num_units + 2].set_xlim(0, hours)
    ax[num_units + 2].set_ylim(0)
    ax[num_units + 2].set_ylabel("Mass of CO2 (kg)")
    ax[num_units + 2].set_xlabel("Time (h)")
    fig.tight_layout()
    plt.show()
