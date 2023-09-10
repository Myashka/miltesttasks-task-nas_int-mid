import os
import torch

import os


def save_checkpoint(
    model, optimizer, epoch, loss, accuracy, checkpoint_dir, prefix="checkpoint"
):
    """
    Save model checkpoint.

    :param model: Model to be saved.
    :type model: torch.nn.Module
    :param optimizer: Optimizer for the model.
    :type optimizer: torch.optim.Optimizer
    :param epoch: Current epoch.
    :type epoch: int
    :param loss: Loss on validation.
    :type loss: float
    :param accuracy: Accuracy on validation.
    :type accuracy: float
    :param checkpoint_dir: Directory where checkpoints are saved.
    :type checkpoint_dir: str
    :param prefix: Prefix for the checkpoint file name.
    :type prefix: str
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

    print(f"Checkpoint saved: {checkpoint_path}")
