#!/bin/bash
wandb online
python system_simulation.py +T=48 kinetics=linear sizing=constant +naming='[kinetics,sizing]'
python system_simulation.py +T=48 kinetics=first_order_t90 sizing=constant +naming='[kinetics,sizing]'
python system_simulation.py +T=48 kinetics=linear sizing=detailed_linear +naming='[kinetics,sizing]'
python system_simulation.py +T=48 kinetics=first_order_t90 sizing=detailed_first_order +naming='[kinetics,sizing]'
wandb offline