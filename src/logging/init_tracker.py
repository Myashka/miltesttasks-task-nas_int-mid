import wandb


def init_tracker(config):
    run = wandb.init(
        project=config["wandb"]["project_name"],
        tags=config["wandb"]["tags"],
        name=config["wandb"]["run_name"],
        config=config,
    )
    return run
