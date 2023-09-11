import os
import torch
import wandb


def save_checkpoint(
    model, optimizer, epoch, loss, accuracy, checkpoint_dir, prefix="checkpoint"
):
    checkpoint_name = f"{prefix}-epoch_{epoch}-loss_{loss:.4f}-acc_{accuracy:.4f}.pt"
    # checkpoint_dir = os.path.join(checkpoint_dir, run_name)
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
