#!/bin/bash
wandb online
python system_simulation.py T=48 kinetics=linear sizing=constant +naming='[kinetics,sizing]'
python system_simulation.py T=48 sizing=constant +naming='[kinetics,sizing]'
python system_simulation.py T=48 sizing=detailed_linear +naming='[kinetics,sizing]'
python system_simulation.py T=48 +naming='[kinetics,sizing]'
wandb offline