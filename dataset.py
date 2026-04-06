import glob
import os
import torch
import json
import logging
import random
import io
from tqdm import tqdm
import librosa
import numpy as np
import sox
from torch.utils.data import Dataset
import random
import numpy as np
from torch.utils.data import Dataset


def _load_scp(scp_path):
    """Parse Kaldi-style scp: each line is ``utt_id wav_path`` or a single wav path."""
    samples = []
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) == 1:
                path = parts[0]
                utt = os.path.splitext(os.path.basename(path))[0]
            else:
                utt, path = parts[0], parts[1]
            samples.append((utt, path))
    return samples


class MultiDomainDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        speech_train_shards_dir,
        music_train_shards_dir,
        sound_train_shards_dir,
        segment_size=240000,
        seed=10086,
        base_domain="music", 
    ):
        self.sr = sample_rate
        self.segment_size = segment_size
        self.seed = seed
        self.base_domain = base_domain
        
        self.speech_dataset = SpeechDataset(
            speech_train_shards_dir,
            dataset_name="speech",
            sr=sample_rate,
            segment_size=segment_size
        )

        self.music_dataset = MusicDataset(
            music_train_shards_dir,
            dataset_name="music",
            sr=sample_rate,
            segment_size=segment_size
        )
        
        self.sound_dataset = SoundDataset(
            sound_train_shards_dir,
            dataset_name="sound",
            sr=sample_rate,
            segment_size=segment_size
        )

        self.speech_size = len(self.speech_dataset)
        self.music_size = len(self.music_dataset)
        self.sound_size = len(self.sound_dataset)

        self.current_indices = []
        self.refresh_epoch(0)

    def _get_base_size(self) -> int:
        sizes = {
            "speech": self.speech_size,
            "music": self.music_size,
            "sound": self.sound_size,
            "min": min(self.speech_size, self.music_size, self.sound_size),
            "max": min(self.speech_size, self.music_size, self.sound_size),
        }
        if self.base_domain not in sizes:
            raise ValueError(f"Unknown base_domain={self.base_domain}, choose from {list(sizes.keys())}")
        return sizes[self.base_domain]

    # -------------------------------------------------------
    def refresh_epoch(self, epoch=0):
        """
        Rebuild indices each epoch with ratio:
        speech: music: sound = 6 : 1 : 1
        """
        base = self._get_base_size()

        n_speech = base * 6
        n_music = base * 1
        n_sound = base * 1

        replace_speech = n_speech > self.speech_size
        speech_raw = np.random.choice(self.speech_size, n_speech, replace=replace_speech)
        speech_idx = [(0, int(i)) for i in speech_raw]
        
        replace_music = n_music > self.music_size
        music_raw = np.random.choice(self.music_size, n_music, replace=replace_music)
        music_idx = [(1, int(i)) for i in music_raw]

        replace_sound = n_sound > self.sound_size
        sound_raw = np.random.choice(self.sound_size, n_sound, replace=replace_sound)
        sound_idx = [(2, int(i)) for i in sound_raw]
 
        self.current_indices = speech_idx + music_idx + sound_idx
        
        random.seed(self.seed + epoch)
        random.shuffle(self.current_indices)

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx):
        ds_flag, real_idx = self.current_indices[idx]
        if ds_flag == 0:
            return self.speech_dataset[real_idx]
        elif ds_flag == 1:
            return self.music_dataset[real_idx]
        elif ds_flag == 2:
            return self.sound_dataset[real_idx]
        else:
            raise RuntimeError(f"Unknown ds_flag: {ds_flag}")
        


class _ScpWavDataset(Dataset):
    """Loads mono wav from scp paths; peak-normalizes with librosa then scales by 0.95."""

    def __init__(self, scp_path, dataset_name, sr=24000, segment_size=240000):
        self.scp_path = scp_path
        self.dataset_name = dataset_name
        self.sr = sr
        self.segment_size = segment_size
        self.samples = _load_scp(scp_path)

    def __len__(self):
        return len(self.samples)

    def process_audio(self, wav, utt):
        wav_length = len(wav)
        if wav_length > self.segment_size:
            start = np.random.randint(0, wav_length - self.segment_size)
            wav = wav[start : start + self.segment_size]
        elif (
            0.8 * self.segment_size < wav_length < self.segment_size
            or wav_length == self.segment_size
        ):
            padding_length = self.segment_size - wav_length
            wav = np.pad(wav, (0, padding_length), mode="constant")
        else:
            repeat = int(np.ceil(self.segment_size / wav_length))
            wav_repeat = np.tile(wav, repeat)
            start = np.random.randint(0, wav_repeat.size - self.segment_size + 1)
            seg_wav = wav_repeat[start : start + self.segment_size]
            return torch.from_numpy(seg_wav.copy()).to(torch.float32), utt

        return torch.from_numpy(wav.copy()).float(), utt

    def __getitem__(self, idx):
        utt, wav_path = self.samples[idx]
        try:
            wav, _ = librosa.load(wav_path, sr=self.sr, mono=True)
        except Exception:
            return None

        wav = librosa.util.normalize(wav) * 0.95
        if wav.size == 0:
            return None
        wav, utt = self.process_audio(wav, utt)
        if wav is None:
            return None
        return {"wav": wav, "utt": utt, "text": None}


class SpeechDataset(_ScpWavDataset):
    def __init__(self, speech_train_shards_dir, dataset_name="speech", sr=24000, segment_size=240000):
        self.speech_train_shards_dir = speech_train_shards_dir
        super().__init__(speech_train_shards_dir, dataset_name, sr, segment_size)


class MusicDataset(_ScpWavDataset):
    def __init__(self, music_train_shards_dir, dataset_name="music", sr=24000, segment_size=240000):
        self.music_train_shards_dir = music_train_shards_dir
        super().__init__(music_train_shards_dir, dataset_name, sr, segment_size)


class SoundDataset(_ScpWavDataset):
    def __init__(self, sound_train_shards_dir, dataset_name="sound", sr=24000, segment_size=240000):
        self.sound_train_shards_dir = sound_train_shards_dir
        super().__init__(sound_train_shards_dir, dataset_name, sr, segment_size)