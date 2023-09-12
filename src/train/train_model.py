from tqdm import tqdm
from collections import defaultdict

from src.utils.helper_functions import accumulate_metrics

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

    for data, target in tqdm(dataloader, desc="Training epoch"):
        loss, logits = train_batch(model, data, target, optimizer, criterion, device)
        batch_size = data.size(0)
        total_loss += loss * batch_size

        if metrics:
            metric_results = accumulate_metrics(logits, target, batch_size, metrics, metric_results)

        if sampler_config and sampler_config.get("batch_mode", False):
            model.sampler(sampler_config)

    metric_results["train_loss"] = total_loss / total_samples
    for k, v in metric_results.items():
        if k != "train_loss":
            metric_results[k] = v / total_samples

    return metric_results