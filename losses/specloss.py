# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
from torchaudio.transforms import MelSpectrogram
import torch
from torch import nn
from torch.nn import functional as F

from modules import pad_for_conv1d


class MelSpectrogramWrapper(nn.Module):
    """Wrapper around MelSpectrogram torchaudio transform providing proper padding
    and additional post-processing including log scaling.

    Args:
        n_mels (int): Number of mel bins.
        n_fft (int): Number of fft.
        hop_length (int): Hop size.
        win_length (int): Window length.
        n_mels (int): Number of mel bins.
        sample_rate (int): Sample rate.
        f_min (float or None): Minimum frequency.
        f_max (float or None): Maximum frequency.
        log (bool): Whether to scale with log.
        normalized (bool): Whether to normalize the melspectrogram.
        floor_level (float): Floor level based on human perception (default=1e-5).
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 256, win_length: tp.Optional[int] = None,
                 n_mels: int = 80, sample_rate: float = 22050, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.n_fft = n_fft
        hop_length = int(hop_length)
        self.hop_length = hop_length
        self.mel_transform = MelSpectrogram(n_mels=n_mels, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
                                            win_length=win_length, f_min=f_min, f_max=f_max, normalized=normalized,
                                            window_fn=torch.hann_window, center=False)
        self.floor_level = floor_level
        self.log = log

    def forward(self, x):
        p = int((self.n_fft - self.hop_length) // 2)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (p, p), "reflect")
        # Make sure that all the frames are full.
        # The combination of `pad_for_conv1d` and the above padding
        # will make the output of size ceil(T / hop).
        x = pad_for_conv1d(x, self.n_fft, self.hop_length)
        self.mel_transform.to(x.device)
        mel_spec = self.mel_transform(x)
        B, C, freqs, frame = mel_spec.shape
        if self.log:
            mel_spec = torch.log10(self.floor_level + mel_spec)
        return mel_spec.reshape(B, C * freqs, frame)


class MelSpectrogramL1Loss(torch.nn.Module):
    """L1 Loss on MelSpectrogram.

    Args:
        sample_rate (int): Sample rate.
        n_fft (int): Number of fft.
        hop_length (int): Hop size.
        win_length (int): Window length.
        n_mels (int): Number of mel bins.
        f_min (float or None): Minimum frequency.
        f_max (float or None): Maximum frequency.
        log (bool): Whether to scale with log.
        normalized (bool): Whether to normalize the melspectrogram.
        floor_level (float): Floor level value based on human perception (default=1e-5).
    """
    def __init__(self, sample_rate: int, n_fft: int = 1024, hop_length: int = 256, win_length: int = 1024,
                 n_mels: int = 80, f_min: float = 0.0, f_max: tp.Optional[float] = None,
                 log: bool = True, normalized: bool = False, floor_level: float = 1e-5):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.melspec = MelSpectrogramWrapper(n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                             n_mels=n_mels, sample_rate=sample_rate, f_min=f_min, f_max=f_max,
                                             log=log, normalized=normalized, floor_level=floor_level)

    def forward(self, x, y):
        self.melspec.to(x.device)
        s_x = self.melspec(x)
        s_y = self.melspec(y)
        return self.l1(s_x, s_y)

class MultiScaleMelSpectrogramLossCB(nn.Module):
    """Multi-Scale spectrogram loss (msspec).

    Args:
        sample_rate (int): Sample rate.
        range_start (int): Power of 2 to use for the first scale.
        range_stop (int): Power of 2 to use for the last scale.
        n_mels (int): Number of mel bins.
        f_min (float): Minimum frequency.
        f_max (float or None): Maximum frequency.
        normalized (bool): Whether to normalize the melspectrogram.
        alphas (bool): Whether to use alphas as coefficients or not.
        floor_level (float): Floor level value based on human perception (default=1e-5).
    """
    def __init__(
            self, 
            sample_rate: int, 
            range_start: int = 6, 
            range_end: int = 11,
            n_mels: int = 64, 
            f_min: float = 0.0, 
            f_max: tp.Optional[float] = None,
            normalized: bool = False, 
            alphas: bool = True, 
            floor_level: float = 1e-5
        ):
        super().__init__()
        l1s = list()
        l2s = list()
        self.alphas = list()
        self.total = 0
        self.normalized = normalized
        for i in range(range_start, range_end):
            l1s.append(
                MelSpectrogramWrapper(
                    n_fft=2 ** i, 
                    hop_length=(2 ** i) / 4, 
                    win_length=2 ** i,
                    n_mels=int(n_mels * 2**(i-range_start)), 
                    sample_rate=sample_rate, 
                    f_min=f_min, 
                    f_max=f_max,
                    log=False, 
                    normalized=normalized, 
                    floor_level=floor_level
                )
            )
            l2s.append(
                MelSpectrogramWrapper(
                    n_fft=2 ** i, 
                    hop_length=(2 ** i) / 4, 
                    win_length=2 ** i,
                    n_mels=int(n_mels * 2**(i-range_start)), 
                    sample_rate=sample_rate, 
                    f_min=f_min, 
                    f_max=f_max,
                    log=True, 
                    normalized=normalized, 
                    floor_level=floor_level
                )
            )
            if alphas:
                self.alphas.append(np.sqrt(2 ** i - 1))
            else:
                self.alphas.append(1)
            self.total += self.alphas[-1] + 1

        self.l1s = nn.ModuleList(l1s)
        self.l2s = nn.ModuleList(l2s)

    def forward(self, x, y):
        loss = 0.0
        self.l1s.to(x.device)
        self.l2s.to(x.device)
        for i in range(len(self.alphas)):
            s_x_1 = self.l1s[i](x)
            s_y_1 = self.l1s[i](y)
            s_x_2 = self.l2s[i](x)
            s_y_2 = self.l2s[i](y)
            loss += F.l1_loss(s_x_1, s_y_1) + self.alphas[i] * F.mse_loss(s_x_2, s_y_2)
        if self.normalized:
            loss = loss / self.total
        return loss


# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


import torch
import torch.nn.functional as F
import torch.nn as nn
from librosa.filters import mel as librosa_mel_fn
from scipy import signal

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
        sample_rate: int,
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
        self.sample_rate = sample_rate

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
            self.sample_rate, 2 * (nf - 1), n_mels, fmin, fmax
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


# Loss functions
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
