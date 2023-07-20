#!/bin/zsh
wandb online
python system_simulation.py +naming='[controller]'
python system_simulation.py +wind_max_only=True +naming='[controller,wind_max_only]'
python system_simulation.py controller=cem +dynamics_model=constant_wind +naming='[controller,dynamics_model]'
python system_simulation.py +T=8750 controller=cem +dynamics_model=known_wind +naming='[controller,dynamics_model]'
wandb offline
