import torch
from tqdm import tqdm
from collections import defaultdict
from src.metrics.compute_metrics import compute_metrics


def eval_epoch(model, dataloader, criterion, device, metrics=None, prefix="val"):
    model.eval()
    total_loss = 0.0
    metric_results = defaultdict(float)
    total_samples = len(dataloader.dataset)

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size

            if metrics:
                metric_vals = compute_metrics(logits.cpu(), target.cpu(), metrics)
                for metric_name, metric_value in metric_vals.items():
                    metric_results[f"{prefix}_{metric_name}"] += (
                        metric_value * batch_size
                    )

    metric_results[f"{prefix}_loss"] = total_loss / total_samples
    if metrics:
        for metric_name in metric_vals:
            metric_results[f"{prefix}_{metric_name}"] /= total_samples

    return metric_results
