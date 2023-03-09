#!/bin/bash
wandb online
python system_simulation.py kinetics=first_order_t90 sizing=detailed_first_order
python system_simulation.py kinetics=first_order_t90 sizing=detailed_first_order +wind_max_only=True
python system_simulation.py kinetics=first_order_t90 sizing=detailed_first_order controller=cem +dynamics_model=constant_wind
python system_simulation.py kinetics=first_order_t90 sizing=detailed_first_order controller=cem +dynamics_model=known_wind
wandb offline
