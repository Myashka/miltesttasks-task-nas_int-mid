import wandb


def init_tracker(config):
    """
    Initialize a Weights & Biases (wandb) run for tracking experiments.

    :param config: Configuration dictionary containing Weights & Biases details.
    :type config: dict
    :rtype: wandb.wandb_run.Run
    :return: An initialized wandb run instance.
    """
    run = wandb.init(
        project=config["wandb"]["project_name"],
        tags=config["wandb"]["tags"],
        name=config["wandb"]["run_name"],
        config=config,
    )
    return run
