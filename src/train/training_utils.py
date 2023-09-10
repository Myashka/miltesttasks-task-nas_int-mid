from tqdm import tqdm
from collections import defaultdict

from src.metrics.compute_metrics import compute_metrics


def train_batch(model, data, target, optimizer, criterion, device):
    model.train()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()

    return loss.item(), logits.cpu().detach()


def train_epoch(model, dataloader, optimizer, criterion, device, metrics=None):
    model.train()
    total_loss = 0.0
    metric_results = defaultdict(float)
    total_samples = len(dataloader.dataset)

    for data, target in tqdm(dataloader, desc="Training epoch"):
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
