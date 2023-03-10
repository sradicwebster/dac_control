#!/bin/bash
wandb online
python system_simulation.py kinetics=first_order_t90 sizing=detailed_first_order +naming='[controller]'
python system_simulation.py kinetics=first_order_t90 sizing=detailed_first_order +wind_max_only=True +naming='[controller,wind_max_only]'
python system_simulation.py kinetics=first_order_t90 sizing=detailed_first_order controller=cem +dynamics_model=constant_wind +naming='[controller]'
python system_simulation.py kinetics=first_order_t90 sizing=detailed_first_order controller=cem +dynamics_model=known_wind +naming='[controller]'
wandb offline
