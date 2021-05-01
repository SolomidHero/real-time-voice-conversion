from .hparams import cfg

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import os


def load_wav(path):
  wav, _ = librosa.load(path, sr=cfg.data.sample_rate)
  wav = librosa.util.normalize(wav) * 0.95

  return wav


def save_wav(path, wav, sr):
  """Save audio to path"""
  wav = np.clip(wav, -1.0, 1.0)
  sf.write(path, wav, sr)


def get_subdirs(dir_path):
  dir_path = Path(dir_path)
  dirnames = [p.stem for p in dir_path.iterdir() if p.is_dir()]
  return dirnames


def has_ext(filepath, ext):
  if isinstance(ext, str):
    return Path(filepath).match(f'*{ext}')
  else:
    return any(Path(filepath).match(f'*{e}') for e in ext)


def get_filepathes(dir_path, ext='.wav'):
  for d_path, _, f_names in os.walk(dir_path):
    for fn in f_names:
      if not has_ext(fn, ext):
        continue

      yield os.path.join(d_path, fn)
