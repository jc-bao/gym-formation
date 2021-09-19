import yaml
import os

def get_config():
    with open(os.path.dirname(__file__)+"/parameters.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config