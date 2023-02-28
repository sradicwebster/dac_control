#!/bin/bash
wandb online
python system_simulation.py
python system_simulation.py controller=cem +dynamics_model=constant_wind
wandb offline
