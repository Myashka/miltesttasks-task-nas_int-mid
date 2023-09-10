import wandb


def wandb_log(results):
    """
    Log results to wandb.

    :param results: A dictionary containing the metrics and values to log.
    :type results: dict
    """
    wandb.log(results)
