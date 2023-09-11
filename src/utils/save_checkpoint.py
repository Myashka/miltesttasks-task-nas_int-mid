import os
import torch
import wandb


def save_checkpoint(
    model, optimizer, epoch, loss, accuracy, checkpoint_dir, prefix="checkpoint"
):
    """
    Save a checkpoint of the model along with optimizer state, epoch, loss, and accuracy.
    Additionally, logs the checkpoint to wandb.

    :param model: Model whose state is to be saved
    :type model: torch.nn.Module
    :param optimizer: Optimizer whose state is to be saved
    :type optimizer: torch.optim.Optimizer
    :param epoch: Current training epoch
    :type epoch: int
    :param loss: Loss value at the current epoch
    :type loss: float
    :param accuracy: Accuracy value at the current epoch
    :type accuracy: float
    :param checkpoint_dir: Directory where checkpoint will be saved
    :type checkpoint_dir: str
    :param prefix: Prefix for the checkpoint filename
    :type prefix: str, optional
    """
    checkpoint_name = f"{prefix}-epoch_{epoch}-loss_{loss:.4f}-acc_{accuracy:.4f}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
        },
        checkpoint_path,
    )

    artifact = wandb.Artifact(
        name=checkpoint_dir.split("/")[-1],
        type="model-checkpoint",
        description="Checkpoint of the model at a given epoch",
        metadata={"epoch": epoch, "loss": loss, "accuracy": accuracy},
    )

    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    print(f"Checkpoint saved: {checkpoint_path}")
