# OmniCodec

OmniCodec: Low Frame Rate Universal Audio Codec with Semantic–Acoustic Disentanglement

- Demo Page: [OmniCodec Demo Page](https://hujingbin1.github.io/OmniCodec-Demo-Page/)
- Huggingface: [Huggingface](https://huggingface.co/ASLP-lab/OmniCodec)
- Arxiv: [Arxiv](https://arxiv.org/html/2603.20638v1)


## Overview

This repo contains:

- **Training**: `train.py` (Accelerate + GAN / WavLM-related losses per config)
- **Dataset**: `dataset.py` (multi-domain mixing; loads audio paths from `scp`)
- **Inference**: `infer.py` (reconstructs audio with a pretrained checkpoint)
- **Config**: `config/config_omnicodec.yaml`

## Environment

### Requirements

Install python dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- `requirements.txt` contains an editable install line `-e OmniCodec/transformers-main`. Make sure the referenced path exists in your environment, or adjust/remove that line if you already have `transformers` installed.

## Data preparation (scp)

The training config expects 3 **scp files** (one per domain): speech / music / sound.

Each line in scp can be either:

- `utt_id /abs/or/rel/path/to/audio.wav`
- `/abs/or/rel/path/to/audio.wav` (utt will be inferred from filename)

Example:

```text
utt0001 /data/speech/utt0001.wav
utt0002 /data/speech/utt0002.wav
```

### What dataset does

For each item, `dataset.py` will:

- load audio with `librosa.load(..., sr=sample_rate, mono=True)`
- apply `librosa.util.normalize(wav) * 0.95`
- crop/pad/repeat to `segment_size` (default: 240000 samples @ 24kHz = 10s)
- return a dict: `{"wav": Tensor[T], "utt": str, "text": None}`

Failed samples return `None` and are filtered by `collate_fn` in `train.py`.

## Configure

Edit `config/config_omnicodec.yaml`:

- **Data**
  - `data.speech_train_shards_dir`: path to `speech.scp`
  - `data.music_train_shards_dir`: path to `music.scp`
  - `data.sound_train_shards_dir`: path to `sound.scp`
  - `data.sample_rate`: default `24000`
  - `data.segment_size`: default `240000`
- **Pretrained SSL (WavLM)**
  - `model.wavlmloss.ckpt_path`: default `pretrain_model/ssl/wavlm-base-plus`
  - `wav_lm_model`: default `pretrain_model/ssl/wavlm_model/wavlm`
- **Output**
  - `train.save_dir`: default `./exps/omnicodec`

## Training

Run training with the provided config:

```bash
python train.py -c config/config_omnicodec.yaml
```

Checkpoints and logs are written to `train.save_dir` (default: `./exps/omnicodec`).

## Inference (reconstruction)

### Prepare checkpoint

`infer.py` loads the checkpoint from:

- `pretrained_model/omnicodec.pth`

Place your pretrained weights at that path (or edit `infer.py` to point to your checkpoint).

### Run

Put test audio files in:

- `./testset/speech/`

Then run:

```bash
python infer.py -c config/config_omnicodec.yaml
```

Outputs will be written to:

- `./outputs/`

## Project structure

```text
.
├─ config/
│  └─ config_omnicodec.yaml
├─ dataset.py
├─ train.py
├─ infer.py
├─ models/
├─ modules/
├─ quantization/
├─ discriminators/
├─ losses/
├─ utils/
└─ requirements.txt
```

## Acknowledgements

This codebase builds on ideas and components from modern neural audio codecs (e.g., SEANet-style backbones, multi-codebook RVQ, adversarial training) and self-supervised perception (e.g., WavLM), as discussed in the [OmniCodec paper](https://arxiv.org/html/2603.20638v1).

## Citation

If you use this work, please cite:

```bibtex
@misc{omnicodec2026,
  title={OmniCodec: Low Frame Rate Universal Audio Codec with Semantic-Acoustic Disentanglement},
  author={Hu, Jingbin and Haoyu, Che and Dake, Guo and Qirui, Li and Wenhao, Xie and Huakang, Wang and Guobin, Xie and Hanke, Zhang and Chengyou, Li and Pengyuan, Xie and Chuan, Xie and Qiang, Lei},
  year={2026},
  eprint={2603.20638},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2603.20638},
}
```

The author list above is taken from the arXiv HTML front matter; if your toolchain requires the exact publisher metadata, please copy the official BibTeX from [arXiv](https://arxiv.org/abs/2603.20638) after it is available.

## License

See the repository license