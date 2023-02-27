#!/bin/bash
wandb online
python system_simulation.py
python system_simulation.py kinetics=first_order_t90
python system_simulation.py unit_sizing=detailed_linear
python system_simulation.py kinetics=first_order_t90 unit_sizing=detailed_first_order
wandb offline
