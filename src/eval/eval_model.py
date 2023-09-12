import torch
from tqdm import tqdm
from collections import defaultdict
from src.utils.helper_functions import handle_sampler, accumulate_metrics


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
    model.eval()
    total_loss = 0.0
    metric_results = defaultdict(float)
    total_samples = len(dataloader.dataset)

    all_predictions = []
    all_targets = []

    handle_sampler(model, sampler_config)

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size

            if metrics:
                metric_results = accumulate_metrics(logits, target, batch_size, metrics, metric_results, prefix)

            if return_outputs:
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())

            handle_sampler(model, sampler_config)

    metric_results[f"{prefix}_loss"] = total_loss / total_samples
    for k, v in metric_results.items():
        if k != f"{prefix}_loss":
            metric_results[k] = v / total_samples

    return (metric_results, all_predictions, all_targets) if return_outputs else metric_results