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
    for name, module in supernet.named_children():
        # if name in ['init_conv', 'variable_block1', 'downsample_conv', 'variable_block2', 'fc']:
        module.load_state_dict(subnet.state_dict())

def train_epoch(
    model, dataloader, optimizer, criterion, device, metrics=None, sampler_config=None
):
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