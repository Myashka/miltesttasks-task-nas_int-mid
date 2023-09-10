import os
import torch
import torch.optim as optim

from src.models.SuperNet import SuperNet


def load_model(config, device):
    model = SuperNet(config.get("model_config"))
    model.to(device)

    optimizer_config = config.get("optimizer", {})
    if optimizer_config:
        optimizer_name = optimizer_config.get("name", "AdamW")
        optimizer_params = optimizer_config.get("params", {})
        optimizer = getattr(optim, optimizer_name)(model.parameters(), **optimizer_params)
    else:
        optimizer = None

    checkpoint_path = config["model_config"].get("checkpoint_path")
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        last_epoch = checkpoint.get("epoch", 0)
        best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
    else:
        last_epoch = 0
        best_val_accuracy = 0.0

    return model, optimizer, last_epoch, best_val_accuracy
