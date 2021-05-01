import pytest
import os
import numpy as np
from utils import load_wav


@pytest.fixture(scope="session")
def example_wav():
  wav = load_wav(
    os.path.join(os.path.dirname(__file__), "../datasets/test/example/example.wav")
  )
  assert len(wav.shape) == 1
  return wav
