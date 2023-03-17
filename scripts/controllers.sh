#!/bin/bash
wandb online
python system_simulation.py +T=48 +naming='[controller]'
python system_simulation.py +T=48 +wind_max_only=True +naming='[controller,wind_max_only]'
python system_simulation.py +T=48 controller=cem +dynamics_model=constant_wind +naming='[controller,dynamics_model]'
python system_simulation.py +T=48 controller=cem +dynamics_model=known_wind +naming='[controller,dynamics_model]'
wandb offline
