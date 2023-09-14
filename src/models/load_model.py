import os
import torch
import torch.optim as optim

from src.models.SuperNet import SuperNet


def load_model(config, device):
    """
    Load a SuperNet model along with its optimizer, if specified. The function also handles the loading of the model's state 
    from a checkpoint if a path is provided and the checkpoint exists.

    :param config: Configuration dictionary containing model and optimizer details.
    :type config: dict
    :param device: Device to which the model should be loaded (e.g., 'cuda' or 'cpu').
    :type device: str or torch.device

    :rtype: tuple
    :return: A tuple containing the following:
             - model: Loaded SuperNet model.
             - optimizer: Loaded optimizer if specified, otherwise None.
             - last_epoch: The last epoch number from the checkpoint or 0 if not available.
             - best_val_accuracy: The best validation accuracy from the checkpoint or 0.0 if not available.
    """
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
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        last_epoch = checkpoint.get("epoch", 0)
        best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
    else:
        last_epoch = 0
        best_val_accuracy = 0.0

    return model, optimizer, last_epoch, best_val_accuracy
