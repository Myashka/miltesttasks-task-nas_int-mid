import torch
import pandas as pd
import wandb
import click

from src.data.make_dataloaders import make_dataloaders
from src.models.load_model import load_model
from src.eval.eval_model import eval_epoch
from src.losses.CS_loss import CS_loss
from src.metrics.accuracy import accuracy
from src.utils import set_random_seed, load_config
from src.logging.wandb_logging import wandb_log


@click.command()
@click.option("--config-file", default="config.yaml", help="Path to config YAML file")
@click.option("--output-file", default="results.csv")
def inference(config_file, output_file):
    config = load_config(config_file)

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

    sampler_config = config.get("sampler_config")

    test_results, all_predictions, all_targets = eval_epoch(model, test_loader, criterion, device, metrics, prefix="test", sampler_config=sampler_config)
    wandb_log(test_results)
    df = pd.DataFrame({"Target": all_targets, "Pred": all_predictions})
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    inference()
