from src.metrics.compute_metrics import compute_metrics


def accumulate_metrics(
    logits, target, batch_size, metrics, metric_results, prefix="train"
):
    """
    Compute and accumulate metrics over batches. This function calculates metrics using the provided logits and targets,
    then accumulates the results in the metric_results dictionary.

    :param logits: Model's output logits.
    :type logits: torch.Tensor
    :param target: Ground truth targets.
    :type target: torch.Tensor
    :param batch_size: Number of samples in the current batch.
    :type batch_size: int
    :param metrics: List of metric functions to compute.
    :type metrics: list
    :param metric_results: Dictionary to accumulate the metric results.
    :type metric_results: dict
    :param prefix: Prefix to add to metric names. Default is 'train'.
    :type prefix: str

    :rtype: dict
    :return: Updated dictionary containing accumulated metrics.
    """
    metric_vals = compute_metrics(logits.cpu(), target.cpu(), metrics)
    for metric_name, metric_value in metric_vals.items():
        metric_results[f"{prefix}_{metric_name}"] += metric_value * batch_size
    return metric_results
