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

def update_weights_in_supernet(supernet, subnet):
    """
    Update the weights of the supernet using the weights from the subnet.

    :param supernet: The main model (SuperNet) whose weights need to be updated.
    :type supernet: torch.nn.Module
    :param subnet: The sub-model (SubNet) whose weights will be used for the update.
    :type subnet: torch.nn.Module
    """
    super_state = supernet.state_dict()
    sub_state = subnet.state_dict()

    for name, weight in sub_state.items():
        if name in super_state:
            super_state[name] = weight

    supernet.load_state_dict(super_state)


def train_epoch(
    model, dataloader, optimizer, criterion, device, metrics=None, sampler_config=None
):
    """
    Train a model for one epoch. This function processes all batches in the given dataloader, computes the loss, updates
    the weights using the subnet's weights, and computes specified metrics.

    :param model: Model to be trained.
    :type model: torch.nn.Module
    :param dataloader: DataLoader instance for the training dataset.
    :type dataloader: torch.utils.data.DataLoader
    :param optimizer: Optimizer to update the model's weights.
    :type optimizer: torch.optim.Optimizer
    :param criterion: Loss function.
    :type criterion: callable
    :param device: Device to which data should be loaded (e.g., 'cuda' or 'cpu').
    :type device: str or torch.device
    :param metrics: List of metric functions to compute during training. Default is None.
    :type metrics: list or None
    :param sampler_config: Configuration for model sampler. Default is None.
    :type sampler_config: dict or None
    
    :rtype: dict
    :return: A dictionary containing computed metrics for the training epoch.
    """
    model.train()
    total_loss = 0.0
    metric_results = defaultdict(float)
    total_samples = len(dataloader.dataset)

    submodel = model.sampler(sampler_config)

    for data, target in tqdm(dataloader, desc="Training epoch"):
        loss, logits = train_batch(submodel, data, target, optimizer, criterion, device)
        batch_size = data.size(0)
        total_loss += loss * batch_size

        update_weights_in_supernet(model, submodel)

        if metrics:
            metric_results = accumulate_metrics(logits, target, batch_size, metrics, metric_results)

        submodel = model.sampler(sampler_config)

    metric_results["train_loss"] = total_loss / total_samples
    for k, v in metric_results.items():
        if k != "train_loss":
            metric_results[k] = v / total_samples

    return metric_results