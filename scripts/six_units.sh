#!/bin/zsh
wandb online
python system_simulation.py dac.num_units=6 +naming='[controller]'
python system_simulation.py dac.num_units=6 +wind_max_only=True +naming='[controller,wind_max_only]'
python system_simulation.py dac.num_units=6 controller=cycling +naming='[controller]'
python system_simulation.py dac.num_units=6  controller=cycling +wind_max_only=True +naming='[controller,wind_max_only]'
wandb offline