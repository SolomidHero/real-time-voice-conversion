from utils.hparams import cfg
from engine import VC, Utterance

import numpy as np
import torch
import pytest

@pytest.fixture(scope='module')
def engine():
  return VC()

@pytest.fixture(scope='module')
def src(example_wav):
  return example_wav

@pytest.fixture(scope='module')
def tgts(example_wav):
  return [
    example_wav,
    example_wav[:len(example_wav) // 2],
  ]


def test_prepare(engine, src, tgts):
  src_features, (tgt_mel, tgt_spk_emb) = engine.prepare(src, tgts)

  assert len(src_features.shape) == 3
  assert len(tgt_mel.shape) == 3
  assert len(tgt_spk_emb.shape) == 2


def test_convert(engine, src, tgts):
  mel = engine.convert(*engine.prepare(src, tgts))

  assert len(mel.shape) == 3


def test_vocode(engine, src):
  src = torch.from_numpy(src).unsqueeze(0)
  mel = engine._get_mel(src)

  wav = engine.vocode(mel)

  assert len(wav.shape) == 2


def test_e2e(engine, src, tgts):
  # using build-in __call__
  src_utt = Utterance(wav=src)
  tgt_utts = [Utterance(wav=wav) for wav in tgts]
  out1 = engine(src_utt, tgt_utts)

  # using step-by-step
  out2 = engine.convert(*engine.prepare(src, tgts))
  out2 = engine.vocode(out2).cpu().numpy()

  assert np.allclose(out1, out2)
