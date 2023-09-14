import torch
from tqdm import tqdm
from collections import defaultdict
from src.utils.helper_functions import accumulate_metrics


def eval_epoch(
    model,
    dataloader,
    criterion,
    device,
    metrics=None,
    prefix="val",
    sampler_config=None,
    return_outputs=False,
):
    """
    Evaluate a model over an entire epoch. This function processes all batches in the given dataloader, computes the loss
    and specified metrics, and optionally returns model outputs.

    :param model: Model to be evaluated.
    :type model: torch.nn.Module
    :param dataloader: DataLoader instance for the evaluation dataset.
    :type dataloader: torch.utils.data.DataLoader
    :param criterion: Loss function.
    :type criterion: callable
    :param device: Device to which data should be loaded (e.g., 'cuda' or 'cpu').
    :type device: str or torch.device
    :param metrics: List of metric functions to compute during evaluation. Default is None.
    :type metrics: list or None
    :param prefix: Prefix to add to metric names. Default is 'val'.
    :type prefix: str
    :param sampler_config: Configuration for model sampler. Default is None.
    :type sampler_config: dict or None
    :param return_outputs: If True, function also returns model outputs and targets. Default is False.
    :type return_outputs: bool

    :rtype: dict or tuple
    :return: A dictionary containing computed metrics. If return_outputs is True, a tuple containing the following is returned:
             - metric_results: Dictionary containing computed metrics.
             - all_predictions: List containing model's predictions.
             - all_targets: List containing ground truth targets.
    """
    model.eval()
    total_loss = 0.0
    metric_results = defaultdict(float)
    total_samples = len(dataloader.dataset)

    all_predictions = []
    all_targets = []

    submodel = model.sampler(sampler_config)

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            logits = submodel(data)
            loss = criterion(logits, target)
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size

            if metrics:
                metric_results = accumulate_metrics(
                    logits, target, batch_size, metrics, metric_results, prefix
                )

            if return_outputs:
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())

            submodel = model.sampler(sampler_config)

    metric_results[f"{prefix}_loss"] = total_loss / total_samples
    for k, v in metric_results.items():
        if k != f"{prefix}_loss":
            metric_results[k] = v / total_samples

    return (
        (metric_results, all_predictions, all_targets)
        if return_outputs
        else metric_results
    )
