import argparse
import wandb
import yaml
import subprocess
import os


def run(config, count, num_proc):
    with open(f"configs/sweep/{config}.yaml", 'r') as file:
        sweep_config = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep_config, project=sweep_config["project"])
    if num_proc == 1:
        wandb.agent(sweep_id, project=sweep_config["project"], count=count)
    else:
        command = ['wandb', 'agent', f'{os.environ["WANDB_ENTITY"]}/{sweep_config["project"]}/{sweep_id}']
        processes = [subprocess.Popen(command) for _ in range(num_proc)]
        for p in processes:
            p.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()
    run(args.config, args.count, args.num_proc)
