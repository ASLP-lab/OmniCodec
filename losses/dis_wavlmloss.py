# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


import torch
import torch.nn.functional as F
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn
from scipy import signal
import torchaudio
from transformers import AutoModel

import typing
from typing import Optional, List, Union, Dict, Tuple
from collections import namedtuple
import math
import functools


# Adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py under the MIT license.
#   LICENSE is in incl_licenses directory.
class MultiScaleMelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [5, 10, 20, 40, 80, 160, 320],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [32, 64, 128, 256, 512, 1024, 2048]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 0.0 (no ampliciation on mag part)
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 1.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    Additional code copied and modified from https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
    """

    def __init__(
        self,
        sampling_rate: int,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None],
        window_type: str = "hann",
    ):
        super().__init__()
        self.sampling_rate = sampling_rate

        STFTParams = namedtuple(
            "STFTParams",
            ["window_length", "hop_length", "window_type", "match_stride"],
        )

        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    @staticmethod
    @functools.lru_cache(None)
    def get_window(
        window_type,
        window_length,
    ):
        return signal.get_window(window_type, window_length)

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(
        self,
        wav,
        n_mels,
        fmin,
        fmax,
        window_length,
        hop_length,
        match_stride,
        window_type,
    ):
        """
        Mirrors AudioSignal.mel_spectrogram used by BigVGAN-v2 training from: 
        https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        """
        B, C, T = wav.shape

        if match_stride:
            assert (
                hop_length == window_length // 4
            ), "For match_stride, hop must equal n_fft // 4"
            right_pad = math.ceil(T / hop_length) * hop_length - T
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0

        wav = torch.nn.functional.pad(wav, (pad, pad + right_pad), mode="reflect")

        window = self.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(wav.device).float()

        stft = torch.stft(
            wav.reshape(-1, T),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft.shape
        stft = stft.reshape(B, C, nf, nt)
        if match_stride:
            """
            Drop first two and last two frames, which are added, because of padding. Now num_frames * hop_length = num_samples.
            """
            stft = stft[..., 2:-2]
        magnitude = torch.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(
            self.sampling_rate, 2 * (nf - 1), n_mels, fmin, fmax
        )
        mel_basis = torch.from_numpy(mel_basis).to(wav.device)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)

        return mel_spectrogram

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """

        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "match_stride": s.match_stride,
                "window_type": s.window_type,
            }

            x_mels = self.mel_spectrogram(x, **kwargs)
            y_mels = self.mel_spectrogram(y, **kwargs)
            x_logmels = torch.log(
                x_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))
            y_logmels = torch.log(
                y_mels.clamp(min=self.clamp_eps).pow(self.pow)
            ) / torch.log(torch.tensor(10.0))

            loss += self.log_weight * self.loss_fn(x_logmels, y_logmels)
            loss += self.mag_weight * self.loss_fn(x_logmels, y_logmels)

        return loss
    

# Adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py under the MIT license.
#   LICENSE is in incl_licenses directory.
class GZMultiScaleMelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [5, 10, 20, 40, 80, 160, 320],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [32, 64, 128, 256, 512, 1024, 2048]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 0.0 (no ampliciation on mag part)
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 1.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    Additional code copied and modified from https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
    """

    def __init__(
        self,
        sampling_rate: int,
        n_mels: List[int] = [160, 320],
        window_lengths: List[int] = [1024, 2048],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0, 0],
        mel_fmax: List[float] = [None, None],
        window_type: str = "hann",
    ):
        super().__init__()
        self.sampling_rate = sampling_rate

        STFTParams = namedtuple(
            "STFTParams",
            ["window_length", "hop_length", "window_type", "match_stride"],
        )

        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow

    @staticmethod
    @functools.lru_cache(None)
    def get_window(
        window_type,
        window_length,
    ):
        return signal.get_window(window_type, window_length)

    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

    def mel_spectrogram(
        self,
        wav,
        n_mels,
        fmin,
        fmax,
        window_length,
        hop_length,
        match_stride,
        window_type,
    ):
        """
        Mirrors AudioSignal.mel_spectrogram used by BigVGAN-v2 training from: 
        https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        """
        B, C, T = wav.shape

        if match_stride:
            assert (
                hop_length == window_length // 4
            ), "For match_stride, hop must equal n_fft // 4"
            right_pad = math.ceil(T / hop_length) * hop_length - T
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0

        wav = torch.nn.functional.pad(wav, (pad, pad + right_pad), mode="reflect")

        window = self.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(wav.device).float()

        stft = torch.stft(
            wav.reshape(-1, T),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft.shape
        stft = stft.reshape(B, C, nf, nt)
        if match_stride:
            """
            Drop first two and last two frames, which are added, because of padding. Now num_frames * hop_length = num_samples.
            """
            stft = stft[..., 2:-2]
        magnitude = torch.abs(stft)

        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(
            self.sampling_rate, 2 * (nf - 1), n_mels, fmin, fmax
        )
        mel_basis = torch.from_numpy(mel_basis).to(wav.device)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)

        return mel_spectrogram

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """

        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "match_stride": s.match_stride,
                "window_type": s.window_type,
            }

            x_mels = self.mel_spectrogram(x, **kwargs)
            y_mels = self.mel_spectrogram(y, **kwargs)
            # x_logmels = torch.log(
            #     x_mels.clamp(min=self.clamp_eps).pow(self.pow)
            # ) / torch.log(torch.tensor(10.0))
            # y_logmels = torch.log(
            #     y_mels.clamp(min=self.clamp_eps).pow(self.pow)
            # ) / torch.log(torch.tensor(10.0))

            # loss += self.log_weight * self.loss_fn(x_logmels, y_logmels)
            # loss += self.mag_weight * self.loss_fn(x_logmels, y_logmels)

            loss += self.loss_fn(x_mels, y_mels) * 10.0

        return loss



def feature_loss(
    fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]
) -> torch.Tensor:

    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2  # This equates to lambda=2.0 for the feature matching loss


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(
    disc_outputs: List[torch.Tensor],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:

    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

class DisWavLMLoss(torch.nn.Module):
    def __init__(self, wd, model='/home/xj_data/guodake.gdk/wav_lm_base_plus'):
        super(DisWavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model, use_safetensors=False)
        self.wd = wd
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False
        self.resampler = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)

    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav = self.resampler(wav.float())
            wav_embeddings = self.wavlm(
                input_values=wav, output_hidden_states=True
            ).hidden_states
        y_rec = self.resampler(y_rec.float())
        y_rec_embeddings = self.wavlm(
            input_values=y_rec, output_hidden_states=True
        ).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))

        return floss.mean()

    def generator(self, y_rec):
        y_rec = self.resampler(y_rec.float())
        y_rec_embeddings = self.wavlm(
            input_values=y_rec, output_hidden_states=True
        ).hidden_states

        y_rec_embeddings = (
            torch.stack(y_rec_embeddings, dim=1)
            .transpose(-1, -2)
            .flatten(start_dim=1, end_dim=2)
        )
        y_df_hat_g = self.wd(y_rec_embeddings)
        loss_gen = torch.mean((1 - y_df_hat_g) ** 2)

        return loss_gen

    def discriminator(self, wav, y_rec):
        with torch.no_grad():
            wav = self.resampler(wav.float())
            wav_embeddings = self.wavlm(
                input_values=wav, output_hidden_states=True
            ).hidden_states
            y_rec = self.resampler(y_rec.float())
            y_rec_embeddings = self.wavlm(
                input_values=y_rec, output_hidden_states=True
            ).hidden_states

            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )
            y_rec_embeddings = (
                torch.stack(y_rec_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)
        y_d_gs = self.wd(y_rec_embeddings)

        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

        r_loss = torch.mean((1 - y_df_hat_r) ** 2)
        g_loss = torch.mean((y_df_hat_g) ** 2)

        loss_disc_f = r_loss + g_loss

        return loss_disc_f.mean()

    def discriminator_forward(self, wav):
        with torch.no_grad():
            wav = self.resampler(wav)
            wav_embeddings = self.wavlm(
                input_values=wav, output_hidden_states=True
            ).hidden_states
            y_embeddings = (
                torch.stack(wav_embeddings, dim=1)
                .transpose(-1, -2)
                .flatten(start_dim=1, end_dim=2)
            )

        y_d_rs = self.wd(y_embeddings)

        return y_d_rs