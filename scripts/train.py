import torch
from src.data.make_dataloaders import make_dataloaders
from src.models.load_model import load_model
from src.train.training_utils import train_epoch
from src.eval.eval_utils import eval_epoch
from src.losses.CS_loss import CS_loss
from src.metrics.accuracy import accuracy
from src.utils import save_checkpoint, set_random_seed
from src.logging.wandb_logging import wandb_log

from tqdm import tqdm
import click
import yaml
from yaml import CLoader
import wandb


@click.command()
@click.option("--config-file", default="config.yaml", help="Path to config YAML file")
def train(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    set_random_seed(config["seed"])

    run = wandb.init(
        project="mil-test",
        tags=["trian"],
        name=config["run_name"],
        config=config,
    )
    device = torch.device(f"{config['device']}" if torch.cuda.is_available() else "cpu")

    criterion = CS_loss()
    metrics = {"accuracy": accuracy}

    model, optim, last_epoch, best_val_accuracy = load_model(config, device)
    wandb.watch(model)

    train_loader, val_loader = make_dataloaders(
        train_batch_size=config["data"]["train_batch_size"],
        val_batch_size=config["data"]["val_batch_size"],
        val_fraction=config["data"]["val_fraction"],
    )

    for global_epoch in tqdm(
        range(last_epoch + 1, config["epochs"] + 1), desc="Epochs"
    ):
        model.sampler(config.get("sampler_config"))

        train_results = train_epoch(
            model, train_loader, optim, criterion, device, metrics
        )
        train_results["epoch"] = global_epoch
        train_results["architecture"] = (
            model.layer_config[0] * 10 + model.layer_config[1]
        )
        wandb_log(train_results)

        val_results = eval_epoch(model, val_loader, criterion, device, metrics)
        val_results["epoch"] = global_epoch
        wandb_log(val_results)

        if val_results["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = val_results["val_accuracy"]
            save_checkpoint(
                model,
                optim,
                global_epoch,
                val_results["val_loss"],
                best_val_accuracy,
                config["checkpoint_dir"],
                prefix="best",
            )

        if global_epoch % config["save_interval"] == 0:
            save_checkpoint(
                model,
                optim,
                global_epoch,
                val_results["val_loss"],
                val_results["val_accuracy"],
                config["checkpoint_dir"],
            )
    wandb.finish()


if __name__ == "__main__":
    train()