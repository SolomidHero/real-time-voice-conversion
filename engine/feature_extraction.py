from resemblyzer import VoiceEncoder
from transformers import Wav2Vec2Model

import torch
from torch import nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
from utils.hparams import cfg


def load_pretrained_spk_emb(device='cpu'):
  """Load speaker embedding model"""

  model = VoiceEncoder().to(device).eval()
  model.requires_grad_(False)
  return model


def load_pretrained_feature_extractor(device='cpu', ckpt_path='facebook/wav2vec2-base-960h'):
  """Load pretrained Wav2Vec model."""

  def extract_features(self, wav, mask):
    # wav2vec has window of 400, so we pad to center windows
    wav = torch.nn.functional.pad(wav.unsqueeze(1), (200, 200), mode='reflect').squeeze(1)
    return [self(wav).last_hidden_state]

  Wav2Vec2Model.extract_features = extract_features # for same behaviour as fairseq.Wav2Vec2Model
  model = Wav2Vec2Model.from_pretrained(ckpt_path).eval()
  model.requires_grad_(False)
  return model


class Wav2Mel(nn.Module):
  def __init__(self, n_fft, hop_length, win_length,
    sample_rate, n_mels, f_min, f_max, preemph
  ):
    super().__init__()

    window = torch.hann_window(win_length).float()
    self.register_buffer("window", window)

    mel_basis = torch.from_numpy(librosa_mel_fn(
      sample_rate, n_fft, n_mels, f_min, f_max
    )).float()
    self.register_buffer("mel_basis", mel_basis)

    preemph_kernel = torch.FloatTensor([[[-preemph, 1]]])
    self.register_buffer("preemph_kernel", preemph_kernel)

    self.n_fft = n_fft
    self.hop_length = hop_length
    self.win_length = win_length
    self.sample_rate = sample_rate
    self.n_mels = n_mels

  def forward(self, wav):
    n_pad = self.n_fft // 2

    while len(wav.shape) < 3:
      wav = wav.unsqueeze(0)

    wav = torch.nn.functional.conv1d(wav, self.preemph_kernel, padding=1)[:, :, :-1]

    wav = F.pad(wav, (n_pad, n_pad), "reflect").squeeze(0)
    spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
      window=self.window, center=False, return_complex=True
    ).abs()

    mel = torch.matmul(self.mel_basis, spec)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))

    return log_mel