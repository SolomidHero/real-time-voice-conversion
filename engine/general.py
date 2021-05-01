from .loading import get_vocoder, get_vc_model
from .feature_extraction import load_pretrained_spk_emb, load_pretrained_feature_extractor, Wav2Mel
from utils.hparams import cfg

from resemblyzer import preprocess_wav
import torch
import numpy as np
import librosa
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class Utterance:
  """Audio with its cached features"""
  wav: np.array = field(repr=False, default=None)
  sr: int = None
  path: str = None
  spk_name: str = None
  mel: np.ndarray = field(repr=False, default=None)
  spk_emb: np.array = field(repr=False, default=None)
  features: np.ndarray = field(repr=False, default=None)


  def clear(self):
    self.wav = None
    self.mel = None
    self.features = None

  def get_id(self):
    if self.path is None or self.spk_name is None:
      return
    return (self.spk_name, Path(self.path).stem)

  def __eq__(self, other):
    return self.get_id() == other.get_id()

  def __hash__(self):
    return hash(self.get_id())


class VC:
  def __init__(self):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.feature_extractor = load_pretrained_feature_extractor(device=self.device)
    self.mel_extractor = Wav2Mel(**cfg.data)
    self.spk_emb_extractor = load_pretrained_spk_emb(device=self.device)

    self.model = get_vc_model().to(self.device)
    self.vocoder = get_vocoder().to(self.device)

  def __call__(self, src: Utterance, tgts: List[Utterance], input_sr: int = cfg.data.sample_rate):
    """Convert source utterance from source speaker to target speaker"""

    # preparation
    src_features, tgt_features = self.prepare(src.wav, [tgt.wav for tgt in tgts], input_sr=input_sr)

    # conversion
    out_mel = self.convert(src_features, tgt_features)

    # vocoding
    out_wav = self.vocode(out_mel)

    return out_wav.cpu().numpy()

  def prepare(self, src_wav, tgt_wavs, input_sr=cfg.data.sample_rate):
    src_wav = torch.from_numpy(src_wav).to(self.device)
    if len(src_wav.shape) == 1:
      src_wav = src_wav.unsqueeze(0)
    tgt_wav = torch.from_numpy(
      np.concatenate(tgt_wavs)
    ).to(self.device).unsqueeze(0)

    src_features = self._get_features(src_wav)
    tgt_spk_emb = self._get_spk_emb(tgt_wavs, input_sr)
    tgt_mel = self.mel_extractor(tgt_wav)

    return src_features, (tgt_mel, tgt_spk_emb)

  def convert(self, src_features, tgt_features):
    tgt_mel, tgt_spk_emb = tgt_features
    with torch.no_grad():
      out_mel, _, _, _ = self.model(src_features, tgt_mel, ref_embs=tgt_spk_emb)
    return out_mel

  def vocode(self, mel):
    with torch.no_grad():
      wav = self.vocoder(mel).squeeze(1)
    return wav

  def _get_mel(self, wav):
    return self.mel_extractor(wav)

  def _get_features(self, wav):
    with torch.no_grad():
      return self.feature_extractor.extract_features(wav, None)[0]

  def _get_spk_emb(self, wavs, sr=None):
    wavs = [preprocess_wav(wav, sr) for wav in wavs]
    cat_wav = np.concatenate(wavs, 0)
    spk_emb = self.spk_emb_extractor.embed_utterance(cat_wav)

    return torch.from_numpy(spk_emb).to(self.device).unsqueeze(0)

  # @staticmethod
  # def preprocess_single_wav(fpath_or_wav: Union[str, Path, np.ndarray], src_sr=None, tgt_sr=None):
  #   # TODO
  #   if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
  #     wav, src_sr = librosa.load(str(fpath_or_wav), sr=None)
  #   else:
  #     wav = fpath_or_wav

  #   # Resample the wav
  #   if src_sr is not None and tgt_sr is not None:
  #     wav = librosa.resample(wav, src_sr, tgt_sr)

  #   return wav