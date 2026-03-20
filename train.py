import logging
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import itertools
import accelerate
import gc
import os
import torch
import torch.nn as nn
from datetime import datetime
from accelerate import DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import HQ_LibriTTS_Dataset, MultiDomainDataset
from models.compression_omnicodec_v4_wavlm_scaling import build_model

from losses import MultiScaleMelSpectrogramLoss, WavLMLoss
from losses import generator_loss, feature_loss, discriminator_loss
from losses import DisWavLMLoss

from discriminators import MultiScaleSTFTDiscriminator, DACGANLoss, WavLMDiscriminator
from utils import utils
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup
import math
torch.backends.cudnn.benchmark = True
global_step = 0
device = None
use_cuda = torch.cuda.is_available()

#TODO:完善训练代码