# from audioread.exceptions import NoBackendError
from engine import VC, Utterance
from gui import GUI
from pathlib import Path
import utils
from utils.hparams import cfg

from time import perf_counter as timer
import traceback
import numpy as np
import torch
import os
import sys
from pathlib import Path
from collections import defaultdict

# Use this directory structure for your datasets, or modify it to fit your needs
recognized_datasets = [
  'test'
]

# Maximum of generated wavs to keep on memory
MAX_WAVS = 15
MAX_TARGET_SAMPLES = 10
MAX_LOADED_SAMPLES = 100


class Toolbox:
  def __init__(self, datasets_root, seed):
    sys.excepthook = self.excepthook
    self.datasets_root = datasets_root
    self.utterances = set()
    self.current_generated = (None, None, None, None) # speaker_name, mel, breaks, wav
    self.speaker_filepathes = defaultdict(set)
    self.audio_ext = {'.wav', '.flac', '.mp3'}
    for datafolder in utils.data.get_subdirs(datasets_root):
      self.load_dataset_info(os.path.join(self.datasets_root, datafolder))

    self.engine = None # type: VC
    self.current_src_utt = None
    self.current_tgt_utts = None
    self.current_tgt_spk = None
    self.loaded_utts = []
    self.conv_utts_list = []
    self.conv_utts_idlist = []
    self.self_record_count = 0

    self.trim_silences = True

    # Initialize the events and the interface
    self.ui = GUI()
    self.reset_ui(seed)
    self.setup_events()
    self.ui.start()

  def excepthook(self, exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)
    self.ui.log("Exception: %s" % exc_value)

  def setup_events(self):
    # Dataset, speaker and utterance selection
    self.ui.browser_load_button.clicked.connect(lambda: self.load_from_browser())
    random_func = lambda level: lambda: self.ui.populate_browser(self.datasets_root, recognized_datasets, level)
    self.ui.random_dataset_button.clicked.connect(random_func(0))
    self.ui.random_speaker_button.clicked.connect(random_func(1))
    self.ui.random_utterance_button.clicked.connect(random_func(2))
    self.ui.dataset_box.currentIndexChanged.connect(random_func(1))
    self.ui.src_spk_box.currentIndexChanged.connect(random_func(2))
    self.ui.tgt_spk_box.currentIndexChanged.connect(random_func(2))

    # Utterance selection
    func = lambda: self.load_from_browser(self.ui.browse_file())
    self.ui.browser_browse_button.clicked.connect(func)
    func = lambda: self.ui.draw_utterance(self.ui.selected_utterance, "current")
    self.ui.utterance_history.currentIndexChanged.connect(func)
    func = lambda: self.ui.play(self.ui.selected_utterance.wav, cfg.data.sample_rate)
    self.ui.play_button.clicked.connect(func)
    self.ui.stop_button.clicked.connect(self.ui.stop)
    self.ui.record_button.clicked.connect(self.record)

    # Audio
    self.ui.setup_audio_devices(cfg.data.sample_rate)

    # Wav playback & save
    func = lambda: self.replay_last_wav()
    self.ui.replay_wav_button.clicked.connect(func)
    func = lambda: self.export_current_wave()
    self.ui.export_wav_button.clicked.connect(func)
    self.ui.wavs_cb.currentIndexChanged.connect(self.set_current_utt)

    # Generation
    func = lambda: self.convert() or self.vocode()
    self.ui.generate_button.clicked.connect(func)
    self.ui.synthesize_button.clicked.connect(self.convert)
    self.ui.vocode_button.clicked.connect(self.vocode)
    self.ui.random_seed_checkbox.clicked.connect(self.update_seed_textbox)

    # UMAP legend
    self.ui.clear_button.clicked.connect(self.clear_utterances)

  def set_current_utt(self, index):
    self.current_src_utt = self.conv_utts_list[index]

  def export_current_wave(self):
    self.ui.save_audio_file(self.current_src_utt, cfg.data.sample_rate)

  def replay_last_wav(self):
    self.ui.play(self.current_src_utt, cfg.data.sample_rate)

  def reset_ui(self, seed):
    self.ui.populate_browser(self.datasets_root, recognized_datasets, 0, True)
    self.ui.populate_gen_options(seed, self.trim_silences)

  def load_from_browser(self, fpath=None):
    if fpath is None:
      fpath = Path(self.datasets_root, self.ui.current_dataset_name, self.ui.current_src_spk, self.ui.current_utterance_name)
      name = str(fpath.relative_to(self.datasets_root))
      speaker_name = self.ui.current_dataset_name + '_' + self.ui.current_src_spk

      # Select the next utterance
      if self.ui.auto_next_checkbox.isChecked():
        self.ui.browser_select_next()
    elif fpath == "":
      return
    else:
      name = fpath.name
      speaker_name = fpath.parent.name


    # Get the wav from the disk. We take the wav with the vocoder/synthesizer format for
    # playback, so as to have a fair comparison with the generated audio
    wav = utils.load_wav(str(fpath))
    self.ui.log("Loaded %s" % name)

    self.add_real_utterance(wav, cfg.data.sample_rate, name, speaker_name)

  def record(self):
    wav = self.ui.record_one(cfg.data.sample_rate, 5)
    if wav is None:
      return
    self.ui.play(wav, cfg.data.sample_rate)
    self.self_record_count += 1

    speaker_name = "user_recorder"
    name = f"{speaker_name}_{self.self_record_count}"
    self.add_real_utterance(wav, cfg.data.sample_rate, name, speaker_name)

  def add_real_utterance(self, wav, sr, path, spk_name):
    if self.engine is None:
      self.init_engine()

    # Compute the mel spectrogram
    mel = self.engine._get_mel(torch.from_numpy(wav))
    self.ui.draw_mel(mel.squeeze(0), "current")

    # Compute the embedding
    embed = self.engine._get_spk_emb([wav], sr=sr)

    # Add the utterance
    utterance = Utterance(
      wav=wav, sr=sr,
      path=path, spk_name=spk_name,
      mel=mel.cpu().numpy().squeeze(0), spk_emb=embed.squeeze(0)
    )
    if utterance not in self.utterances:
      self.utterances.add(utterance)
      self.ui.register_utterance(utterance)

    # Plot it
    # self.ui.draw_embed(embed, Path(path).stem, "current")
    self.ui.draw_umap_projections(self.utterances)

  def clear_utterances(self):
    self.utterances.clear()
    self.ui.draw_umap_projections(self.utterances)

  def convert(self):
    self.ui.log("Converting from source to target...")
    self.ui.set_loading(1)

    # Update the synthesizer random seed
    if self.ui.random_seed_checkbox.isChecked():
      seed = int(self.ui.seed_textbox.text())
      self.ui.populate_gen_options(seed, self.trim_silences)
    else:
      seed = None

    tgt_spk = self.ui.current_tgt_spk

    # Synthesize the spectrogram
    if self.engine is None:
      self.init_engine()

    src_wav = self.ui.selected_utterance.wav
    if self.current_tgt_spk is None or self.current_tgt_spk != tgt_spk:
      self.current_tgt_utts = self.get_spk_utterances(tgt_spk)

    tgt_wavs = [tgt.wav for tgt in self.current_tgt_utts]
    prep_data = self.engine.prepare(src_wav, tgt_wavs)
    mel = self.engine.convert(*prep_data)

    self.ui.draw_mel(mel.cpu().numpy().squeeze(0), "converted mel")
    self.current_generated = (self.ui.selected_utterance.spk_name, Path(self.ui.selected_utterance.path).stem, self.ui.current_tgt_spk, mel)
    self.ui.set_loading(0)

  def vocode(self):
    src_spk, basename, tgt_spk, mel = self.current_generated
    assert mel is not None

    # Synthesize the waveform
    if not self.engine:
      self.init_engine()

    # def vocoder_progress(i, seq_len, b_size, gen_rate):
    #   real_time_factor = (gen_rate / cfg.data.sample_rate) * 1000
    #   line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
    #        % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
    #   self.ui.log(line, "overwrite")
    #   self.ui.set_loading(i, seq_len)

    # wav = vocoder.infer_waveform(mel, progress_callback=vocoder_progress)
    wav = self.engine.vocode(mel).squeeze(0).cpu().numpy()
    self.ui.set_loading(0)
    self.ui.log("Done!", "append")


    # Play it
    wav = (wav / np.abs(wav).max()) * 0.95
    self.ui.play(wav, cfg.data.sample_rate)

    # Name it (history displayed in combobox)
    name = f"{src_spk}_to_{tgt_spk}_{basename}"
    spk_name = f"{src_spk}_to_{tgt_spk}"

    # Update wavs combobox
    if len(self.conv_utts_list) > MAX_WAVS:
      self.conv_utts_list.pop()
      self.conv_utts_idlist.pop()
    self.conv_utts_list.insert(0, wav)
    self.conv_utts_idlist.insert(0, name)

    # self.ui.wavs_cb.disconnect()
    self.ui.wavs_cb_model.setStringList(self.conv_utts_idlist)
    self.ui.wavs_cb.setCurrentIndex(0)
    self.ui.wavs_cb.currentIndexChanged.connect(self.set_current_utt)

    # Update current wav
    self.set_current_utt(0)

    # Enable replay and save buttons:
    self.ui.replay_wav_button.setDisabled(False)
    self.ui.export_wav_button.setDisabled(False)

    # Compute speaker embedding
    embed = self.engine._get_spk_emb([wav], sr=cfg.data.sample_rate)

    # Add the utterance
    utterance = Utterance(
      wav=wav, sr=cfg.data.sample_rate,
      path=name, spk_name=spk_name,
      mel=mel.cpu().numpy().squeeze(0), spk_emb=embed.squeeze(0)
    )
    self.utterances.add(utterance)

    # Plot it
    # self.ui.draw_embed(embed, name, "generated")
    self.ui.draw_umap_projections(self.utterances)


  def get_spk_utterances(self, spk_name):
    utts = list(filter(lambda u: u.spk_name == spk_name, self.loaded_utts))
    if len(utts) >= MAX_TARGET_SAMPLES:
      return utts

    utts_pathes = set(map(lambda u: u.path, utts))
    available_utts_pathes = list(filter(lambda p: p not in utts_pathes, self.speaker_filepathes[spk_name]))
    available_utts_pathes = available_utts_pathes[:MAX_TARGET_SAMPLES - len(utts_pathes)]

    new_utts = list(map(lambda p: self.load_utterance(spk_name, p), available_utts_pathes))
    self.loaded_utts.extend(new_utts)
    self.loaded_utts = self.loaded_utts[-MAX_LOADED_SAMPLES:]

    utts.extend(new_utts)
    return utts

  def load_utterance(self, spk_name, path):
    wav = utils.load_wav(path)
    return Utterance(wav, cfg.data.sample_rate, path=path, spk_name=spk_name)

  def load_dataset_info(self, dataset_path):
    speakers = utils.data.get_subdirs(dataset_path)

    for spk in speakers:
      self.speaker_filepathes[spk] = {
        *self.speaker_filepathes[spk],
        *utils.data.get_filepathes(os.path.join(dataset_path, spk), self.audio_ext)
      }

  def init_engine(self):
    self.ui.log("Creating voice conversion model...")
    self.ui.set_loading(1)
    start = timer()
    self.engine = VC()
    self.ui.log("Done (%dms)." % int(1000 * (timer() - start)), "append")
    self.ui.set_loading(0)

  def update_seed_textbox(self):
    self.ui.update_seed_textbox()