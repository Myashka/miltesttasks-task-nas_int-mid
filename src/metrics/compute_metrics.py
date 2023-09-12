def compute_metrics(logits, targets, metrics):
    """
    Compute metrics for the given logits and targets.

    :param logits: Raw predictions from the model.
    :type logits: torch.Tensor
    :param targets: Ground truth labels.
    :type targets: torch.Tensor
    :param metrics: A dictionary of metric functions to compute.
    :type metrics: dict[str, callable]
    :rtype: dict[str, float]
    :return: A dictionary containing the computed metrics.
    """
    results = {}
    for metric_name, metric_fn in metrics.items():
        results[metric_name] = metric_fn(logits, targets)
    return results