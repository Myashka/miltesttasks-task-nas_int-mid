from tqdm import tqdm
from collections import defaultdict

from src.metrics.compute_metrics import compute_metrics


def train_batch(model, data, target, optimizer, criterion, device):
    """
    Train the model on a single batch of data.

    :param model: The model to be trained
    :type model: torch.nn.Module
    :param data: Batch of input data
    :type data: torch.Tensor
    :param target: True labels for the input data
    :type target: torch.Tensor
    :param optimizer: Optimizer for the training
    :type optimizer: torch.optim.Optimizer
    :param criterion: Loss function
    :type criterion: callable
    :param device: Device to which data should be transferred before training (e.g., 'cuda' or 'cpu')
    :type device: str
    :rtype: tuple
    :return: Loss value and logits predicted by the model for this batch
    """
    model.train()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

    return loss.item(), logits.cpu().detach()


def train_epoch(
    model, dataloader, optimizer, criterion, device, metrics=None, sampler_config=None
):
    """
    Train the model for an entire epoch.

    :param model: The model to be trained
    :type model: torch.nn.Module
    :param dataloader: DataLoader providing batches of training data
    :type dataloader: torch.utils.data.DataLoader
    :param optimizer: Optimizer for the training
    :type optimizer: torch.optim.Optimizer
    :param criterion: Loss function
    :type criterion: callable
    :param device: Device to which data should be transferred before training (e.g., 'cuda' or 'cpu')
    :type device: str
    :param metrics: Dictionary of metric functions to compute
    :type metrics: dict, optional
    :param sampler_config: Configuration for the model sampler (if any)
    :type sampler_config: dict, optional
    :rtype: dict
    :return: Dictionary containing the training loss and other computed metrics for the epoch
    """
    model.train()
    total_loss = 0.0
    metric_results = defaultdict(float)
    total_samples = len(dataloader.dataset)

    for data, target in tqdm(
        dataloader, position=1, desc="Training epoch", leave=False
    ):
        model.sampler(sampler_config)
        loss, logits = train_batch(model, data, target, optimizer, criterion, device)
        batch_size = data.size(0)
        total_loss += loss * batch_size

        if metrics:
            metric_vals = compute_metrics(logits, target.cpu(), metrics)
            for metric_name, metric_value in metric_vals.items():
                metric_results[f"train_{metric_name}"] += metric_value * data.size(0)

    metric_results["train_loss"] = total_loss / total_samples
    for metric_name in metric_vals:
        metric_results[f"train_{metric_name}"] /= total_samples

    return metric_results
