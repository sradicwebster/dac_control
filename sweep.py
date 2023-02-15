import argparse
import wandb
import yaml


def run(config, count):
    with open(f"configs/sweep/{config}.yaml", 'r') as file:
        sweep_config = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep_config, project=sweep_config["project"])
    wandb.agent(sweep_id, project=sweep_config["project"], count=count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--count", type=int, default=None)
    args = parser.parse_args()
    run(args.config, args.count)
