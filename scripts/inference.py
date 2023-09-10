import torch
import pandas as pd
import wandb
import click
import yaml
from yaml import CLoader

from src.data.make_dataloaders import make_dataloaders
from src.models.load_model import load_model
from src.eval.eval_utils import eval_epoch
from src.losses.CS_loss import CS_loss
from src.metrics.accuracy import accuracy
from src.utils import set_random_seed
from src.logging.wandb_logging import wandb_log


@click.command()
@click.option("--config-file", default="config.yaml", help="Path to config YAML file")
@click.option("--output-file", default="results.csv")
def inference(config_file, output_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)

    set_random_seed(config["seed"])

    run = wandb.init(
        project="mil-test",
        tags=["test"],
        name=config["run_name"],
        config=config,
    )
    device = torch.device(f"{config['device']}" if torch.cuda.is_available() else "cpu")

    criterion = CS_loss()
    metrics = {"accuracy": accuracy}

    model, _, _, _ = load_model(config, device)

    test_loader = make_dataloaders(
        do_train=False, test_batch_size=config["data"]["test_batch_size"]
    )

    model.sampler(config.get("sampler_config"))

    test_results = eval_epoch(
        model, test_loader, criterion, device, metrics, prefix="test"
    )
    wandb_log(test_results)
    df = pd.DataFrame([test_results])
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    inference()
