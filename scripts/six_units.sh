#!/bin/zsh
wandb online
python system_simulation.py dac.num_units=6 sizing=detailed_fo_co2_total controller=cycling +naming='[controller]'
python system_simulation.py dac.num_units=6 sizing=detailed_fo_co2_total controller=cycling +wind_max_only=True +naming='[controller,wind_max_only]'
python system_simulation.py dac.num_units=6 sizing=detailed_fo_co2_total controller=cem +dynamics_model=constant_wind +naming='[controller,dynamics_model]'
wandb offline
