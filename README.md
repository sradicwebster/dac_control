# Wind-powered direct air capture modelling and control

Project carried out as a visiting doctoral student with the Sustainable Systems Design Lab at the University of Victoria, Canada, to apply machine learning for the optimisation of a offshore direct air capture (DAC) system. See the [Wiki](https://github.com/sradicwebster/dac_control/wiki) for modelling, control algorithms and results.

First, create a conda environment and install the required packages:

```zsh
conda create -n dac_control
conda activate dac_control
pip install -r requirements.txt
```

[Weights and Biases](https://docs.wandb.ai/) is used to track the experiments. After signing up for a free account, run the following:

```zsh
wandb login
wandb online
```

[Hydra](https://hydra.cc/docs/intro/) is used to manage the experiment configuration using [config](configs/) files. To run an experiment with the default configuration (as stated in [main.yaml](configs/main.yaml)), run:

```zsh
python system_simulation.py
```

The experiment configuration can be recorded in a yaml file and then specified when running the script. For example:

```zsh
python system_simulation.py controller=loading kinetics=first_order_t90
```

Alternatively, the configuration can be overriden from the command line. For example:

```zsh
python system_simulation.py dac.num_units=4
```

To specify the cross entropy method (CEM) model predictive controller (MPC), a dynamics model must be specified in the command line, for example:

```zsh
python system_simulation.py controller=cem +dynamics_model=constant_wind
```

A seperate script can be used to train and save a reinforcement learning agent using [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html). 

```zsh
python train_rl_agent.py controller=ppo wind.file="wind_power_train"
```

After training the agent, the learnt policy can be evaluated by running:

```zsh
python system_simulation.py controller=ppo +wandb_name=WANDB_NAME
```