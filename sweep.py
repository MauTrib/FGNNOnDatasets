import wandb
import argparse
import yaml


def change_config(config, sweep_config):
    """This function takes all keys from the sweep_config and changes the config accordingly.
    As config can be nested, we need to recursively go through the config dict.
    """
    changed_config = dict()
    for key, value in config.items():
        if isinstance(value, dict):
            changed_config[key] = change_config(value, sweep_config)
        elif key in sweep_config.keys():
            changed_config[key] = sweep_config[key]
        else:
            changed_config[key] = value
    return changed_config


def sweep(config, sweep_config):
    """This function creates a W&B sweep and launches it."""

    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project=config["project"] + "-sweep")

    def train_sweep(config):
        from commander import train

        wandb.init()
        changed_config = change_config(config, wandb.config)
        train(changed_config)

    # Launch the sweep agent
    wandb.agent(sweep_id, function=lambda: train_sweep(config))


def main():
    parser = argparse.ArgumentParser(description="Main commander for this project.")
    parser.add_argument(
        "--config",
        type=str,
        default="default_config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--sweepconfig",
        type=str,
        default="sweep_config.yaml",
        help="Path to the sweep config file.",
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.sweepconfig, "r") as f:
        sweepconfig = yaml.safe_load(f)

    sweep(config, sweepconfig)


if __name__ == "__main__":
    main()
