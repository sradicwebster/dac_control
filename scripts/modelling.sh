#!/bin/bash
wandb online
python system_simulation.py +naming='[kinetics,sizing]'
python system_simulation.py kinetics=first_order_t90 +naming='[kinetics,sizing]'
python system_simulation.py sizing=detailed_linear +naming='[kinetics,sizing]'
python system_simulation.py kinetics=first_order_t90 sizing=detailed_first_order +naming='[kinetics,sizing]'
wandb offline