import matplotlib.pyplot as plt
import numpy as np
from battery import Battery
from dac import DirectAirCapture
from controllers import RuleMaxCapacity


def run():
    hours = 48
    dt = 10
    num_units = 2

    battery = Battery(capacity=15000,
                      power_max=5000,
                      charge_eff=0.9,
                      discharge_eff=0.9,
                      soc_min=0.1,
                      soc_max=1.0,
                      dt=dt,
                      )
    dac = DirectAirCapture(num_units=num_units,
                           adsorbent_capacity=10000,
                           adsorption_power=5000,
                           desorption_power=10000,
                           adsorption_rate=0.5,
                           desorption_rate=1,
                           dt=dt,
                           )
    wind_power_series = np.load("../data/wind_power_dt10_var500.npy").reshape(-1, 1)

    constants = np.array([dac.adsorbent_capacity,
                          dac.adsorption_power,
                          dac.desorption_power,
                          battery.soc_min,
                          battery.soc_max,
                          battery.power_max,
                          ])
    state = np.concatenate((wind_power_series[0],
                            battery.reset(),
                            dac.reset(),
                            np.zeros(num_units)
                            ))
    rule = RuleMaxCapacity(constants, dt, num_units, 0.1, 0.9, uncertainty_adjustment=0)

    iters = int(hours * 60 / dt)
    all_states = np.zeros((iters+1, len(state)))
    all_states[0] = state
    all_actions = np.zeros((iters, num_units + 1))
    all_rewards = np.zeros((iters, 1))
    for i in range(iters):
        controls = rule.policy(state)
        battery_power = np.array([controls[0]])
        dac_power = controls[1:]
        wind_power = wind_power_series[i+1]
        power_deficit = np.abs(dac_power).sum() - wind_power - battery_power
        if power_deficit > 0:
            # print(f'Step {i}:')
            # if power deficit, reduce charging power and/or increase battery discharge power
            # (subject to max power)
            battery_power = min(battery.discharge_power(), np.abs(dac_power).sum() - wind_power)
            # print(f"Battery power adjusted")
            # if still in deficit, reduce dac power requirement
            power_deficit = np.abs(dac_power).sum() - wind_power - battery_power
            unit = 0
            dac_operation = np.sign(dac_power)
            dac_power = np.abs(dac_power)
            while power_deficit > 0.01:
                dac_power[unit] -= min(dac_power[unit], power_deficit)
                power_deficit = dac_power.sum() - wind_power - battery_power
                unit += 1
            dac_power = dac_operation * dac_power

        assert wind_power + battery_power - np.abs(dac_power).sum() >= 0,\
            f"Power balance not correct as step {i}"

        next_state = np.concatenate((wind_power,
                                     battery.step(battery_power),
                                     dac.step(dac_power),
                                     dac_power
                                     ))
        co2_stored = np.maximum(state[2:] - next_state[2:], 0).sum()
        all_states[i+1] = next_state
        all_actions[i] = np.concatenate((battery_power, dac_power))
        all_rewards[i] = co2_stored
        state = next_state

    # wind, battery and dac plots
    fig, ax = plt.subplots(num_units + 3, figsize=(12, 8))
    ax[0].set_title("Wind power")
    ax[0].plot(np.linspace(0, hours, iters + 1), all_states[:, 0], label='Wind', c='g')
    ax[0].set_xlim(0, hours)
    ax[0].set_ylim(0, np.max(all_states[:, 0]) + 1000)
    ax[0].set_ylabel("Power (kW)")
    ax[1].set_title("Battery")
    ax[1].plot(np.linspace(0, hours, iters, endpoint=False), all_actions[:, 0], label='Power',
               c='b')
    ax[1].set_xlim(0, hours)
    ax[1].set_ylim(np.min(all_actions[:, 0]) - 1000, np.max(all_actions[:, 0]) + 1000)
    ax[1].set_ylabel("Power (kW)")
    ax[1].legend(loc='upper left')
    ax1r = ax[1].twinx()
    ax1r.plot(np.linspace(0, hours, iters + 1), all_states[:, 1] / battery.capacity, label='SOC',
              c='r')
    ax1r.set_ylim(0, 1.3)
    ax1r.set_ylabel("SOC")
    ax1r.legend(loc='upper right')
    for i in range(num_units):
        ax[i + 2].set_title(f"DAC #{i + 1}")
        ax[i + 2].plot(np.linspace(0, hours, iters, endpoint=False), all_actions[:, i + 1],
                       label='Power', c='b')
        ax[i + 2].set_xlim(0, hours)
        ax[i + 2].set_ylim(np.min(all_actions[:, i + 1]) - 1000,
                           np.max(all_actions[:, i + 1]) + 5000)
        ax[i + 2].set_ylabel("Power (kW)")
        ax[i + 2].legend(loc='upper left')
        ax2r = ax[i + 2].twinx()
        ax2r.plot(np.linspace(0, hours, iters + 1), all_states[:, i + 2] / dac.adsorbent_capacity,
                  label='Adsorbent', c='r')
        ax2r.set_ylim(0, 1.3)
        ax2r.set_ylabel("Absorbent capacity")
        ax2r.legend(loc='upper right')
    ax[num_units + 2].set_title("CO2 captured")
    ax[num_units + 2].plot(np.linspace(0, hours, iters, endpoint=False), all_rewards)
    ax[num_units + 2].set_xlim(0, hours)
    ax[num_units + 2].set_ylim(0, np.max(all_rewards))
    ax[num_units + 2].set_ylabel("Mass of CO2 (kg)")
    ax[num_units + 2].set_xlabel("Time (h)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
