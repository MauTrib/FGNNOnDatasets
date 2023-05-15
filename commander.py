import argparse
import yaml
import pytorch_lightning as pl

from data import get_loaders
from models import configure_pipeline
from sweep import sweep
import utils

"""Main code. This is the entry point of the program. It parses the command line arguments and calls the appropriate functions.
We need the functions:
 - train : launches the training of the model
 - test : launches the testing of the model
"""


def get_callbacks(config):
    """Returns wanted callbacks."""
    l_callbacks = [
        pl.callbacks.ModelCheckpoint(monitor="train_loss", save_top_k=1, verbose=True),
        pl.callbacks.EarlyStopping(
            "lr",
            verbose=True,
            mode="max",
            patience=1 + config["train"]["max_epochs"],
            divergence_threshold=config["train"]["optim"]["scheduler_min_lr"],
        ),
    ]
    return l_callbacks


def get_trainer(config):
    """This function configures the trainer. It takes the config file as input and returns a pytorch_lightning.Trainer object."""
    # Base trainer config
    trainer_config = config["train"]

    # Accelerator
    accelerator = utils.get_accelerator_dict(config["device"])
    trainer_config.update(accelerator)

    # Callbacks:
    callbacks = get_callbacks(config)
    trainer_config["callbacks"] = callbacks

    # Clean the config to keep only relevant arguments
    clean_config = utils.restrict_dict_to_function(pl.Trainer.__init__, trainer_config)

    # Add wandb logger
    clean_config["logger"] = pl.loggers.WandbLogger(
        project=config["project"], save_dir="observers/"
    )

    # Create the trainer
    trainer = pl.Trainer(**clean_config)
    return trainer


def train(config, test=False):
    """This function loads the model, the data and the trainer and launches the training."""

    # Load the pipeline (PL wrapper to the model)
    pipeline = configure_pipeline(config)

    # Load the data
    data_train, data_val, data_test = get_loaders(config)

    # Load the trainer
    trainer = get_trainer(config)

    # Launch the training
    trainer.fit(pipeline, data_train, data_val)

    if test:
        trainer.test(ckpt_path="best", dataloaders=data_test)


def test_only(config):
    raise NotImplementedError()


def main():
    """Here, we parse the arguments from the command line. There needs to be:
    - --command argument, launches training or testing (can only be [train, test])
    - --config argument (optional), default is 'default_config.yaml'
    """
    parser = argparse.ArgumentParser(description="Main commander for this project.")
    parser.add_argument(
        "command",
        type=str,
        choices=["train", "test", "traintest", "sweep"],
        help="Command to execute. Can be [train, test]",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default_config.yaml",
        help="Path to the config file.",
    )
    parser.add_argument("--seed", type=int, default=1922409, help="random seed")
    parser.add_argument(
        "--sweepconfig",
        type=str,
        default="sweep_config.yaml",
        help="Path to the sweep config file.",
    )

    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # Use the yaml library to load the config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    try:
        with open(args.sweepconfig, "r") as f:
            sweepconfig = yaml.safe_load(f)
    except FileNotFoundError as fe:
        if args.command == "sweep":
            raise fe

    if args.command[:5] == "train":
        print("Training...")
        train(config, test=(args.command == "traintest"))
    elif args.command == "test":
        print("Testing...")
        test_only(config)
    elif args.command == "sweep":
        print("Sweeping...")
        sweep(config, sweepconfig)


if __name__ == "__main__":
    print("Hello world!")
    main()
