from src.metrics.compute_metrics import compute_metrics

def accumulate_metrics(
    logits, target, batch_size, metrics, metric_results, prefix="train"
):
    metric_vals = compute_metrics(logits.cpu(), target.cpu(), metrics)
    for metric_name, metric_value in metric_vals.items():
        metric_results[f"{prefix}_{metric_name}"] += metric_value * batch_size
    return metric_results
