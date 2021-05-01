import yaml
from omegaconf import DictConfig


with open('config.yaml', 'r') as f:
  cfg = DictConfig(yaml.load(f, Loader=yaml.FullLoader))