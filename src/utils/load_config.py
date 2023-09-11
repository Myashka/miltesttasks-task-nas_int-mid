import yaml
from yaml import CLoader


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=CLoader)
    return config
