from omegaconf import DictConfig
from utils.hparams import cfg
from .vocoder import HifiGenerator

import os
import torch
import json

ckpt_dir = os.path.join(cfg.root_dir, cfg.ckpt_default_path)
ckpt_dict = cfg.ckpt_dict


def get_vocoder():
  group_name = 'hifi_gan'
  download_group(group_name)

  config_path = os.path.join(ckpt_dir, group_name, "config.yaml")
  json_config = json.loads(open(config_path).read())
  with torch.no_grad():
    generator = HifiGenerator(DictConfig(json_config)).eval()

  ckpt_path = os.path.join(ckpt_dir, group_name, "generator")
  state = torch.load(ckpt_path, map_location=torch.device('cpu'))
  generator.load_state_dict(state['generator'])
  generator.remove_weight_norm()

  return generator


def get_vc_model():
  group_name = 'fragmentvc'
  download_group(group_name)

  ckpt_path = os.path.join(ckpt_dir, group_name, "model.pt")
  model = torch.jit.load(ckpt_path).eval()

  return model


def download_group(group_name):
  for filename, (url, agent) in ckpt_dict[group_name].items():
    filepath = os.path.join(ckpt_dir, group_name, filename)
    _download(filepath, url, agent=agent)


def _download(filepath, url, refresh=False, agent='wget'):
  '''
  Download from url into filepath using agent if needed
  Ref: https://github.com/s3prl/s3prl
  '''

  dirpath = os.path.dirname(filepath)
  os.makedirs(dirpath, exist_ok=True)

  if not os.path.isfile(filepath) or refresh:
    if agent == 'wget':
      os.system(f'wget {url} -O {filepath}')
    elif agent == 'gdown':
      import gdown
      gdown.download(url, filepath, use_cookies=False)
    else:
      print('[Download] - Unknown download agent. Only \'wget\' and \'gdown\' are supported.')
      raise NotImplementedError
  else:
    print(f'Using checkpoint found in {filepath}')
