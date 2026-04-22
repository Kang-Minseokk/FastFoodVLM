import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config