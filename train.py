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
from dataset import MultiDomainDataset
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


class WarmupCosineScheduler:
    """封装warmup + cosine，使用transformers库实现，支持checkpoint"""
    def __init__(self, optimizer, total_steps, warmup_steps=2000):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        # 使用transformers的cosine调度器（内置warmup）
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=1  # 单个余弦周期
        )
    
    def step(self):
        self.scheduler.step()
    
    def get_last_lr(self):
        """获取当前学习率"""
        return self.scheduler.get_last_lr()
    
    def state_dict(self):
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)

def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def collate_fn(batch):
    """
    过滤 None 项，堆叠 wav
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    utts = [b["utt"] for b in batch if b["utt"] is not None]
    wavs = [b["wav"] for b in batch if b["wav"] is not None]

    if len(wavs) == 0:
        return None

    wavs = torch.stack(wavs)
    
    return {
        "utt": utts,
        "wav": wavs,
    }
    
    
def main():
    """Assume Single Node Multi GPUs Training Only"""
    hps = utils.get_hparams()
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=hps.train.gradient_accumulation_steps,
        kwargs_handlers=[kwargs]
    )
    accelerator.wait_for_everyone()
    run(accelerator, hps)
    
def run(accelerator: accelerate.Accelerator, hps: utils.HParams):
    global global_step, device
    if accelerator.is_main_process:
        logger = utils.get_logger(hps.train.save_dir)
        logger.info(hps.train)
        logger.info(hps.data)
        logger.info(hps.model)
        utils.check_git_hash(hps.train.save_dir)
        writer = SummaryWriter(log_dir=hps.train.save_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.train.save_dir, "eval"))
    
    torch.manual_seed(hps.train.seed)
    dataset = MultiDomainDataset(**hps.data)

    net_g = build_model(
       hps.model.sample_rate, 
       hps.model.frame_rate,
       hps.model.q_dimension,
       hps.model.semantic_dimension,
       hps.model.seanet,
       hps.model.transformer,
       hps.model.semantic_quantizer,
       hps.model.acoustic_quantizer,
    )
    
    msspec = MultiScaleMelSpectrogramLoss(
       **hps.model.msspec
    )
  
    net_msstftd = MultiScaleSTFTDiscriminator(
       **hps.model.msstftd
    )
    
    net_wld = WavLMDiscriminator(
        **hps.model.wavlmdis
    )
    
    net_dacd = DACGANLoss(
        **hps.model.dacd
    )
    
    net_d = nn.ModuleDict({
        "net_msstftd": net_msstftd,
        "net_wld": net_wld,
        "net_dacd": net_dacd,
    })

    wl_loss = DisWavLMLoss(
        net_wld,
        hps.wav_lm_model,
    )
    
    wavlm = WavLMLoss(
        **hps.model.wavlmloss
    )
    
    params_g = []
    for name, param in net_g.named_parameters():
        if 'encoder_transformer' in name:
            params_g.append({
                'params': param,
                'weight_decay': 5e-2,
                'name': name
            })
        elif 'decoder_transformer' in name:
            params_g.append({
                'params': param,
                'weight_decay': 5e-2, 
                'name': name
            })
        else:
            params_g.append({
                'params': param,
                'weight_decay': 0.0, 
                'name': name
            })
            
    optim_g = torch.optim.AdamW(
        params_g,
        lr=hps.train.learning_rate, 
        betas=(0.5, 0.9), 
        eps=1e-8
    )

    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        lr=hps.train.learning_rate,
        betas=(0.5, 0.9),
        eps=1e-8,
        weight_decay=0.0  
    )

    scheduler_g = WarmupCosineScheduler(optim_g, hps.train.training_steps, warmup_steps=1000)
    scheduler_d = WarmupCosineScheduler(optim_d, hps.train.training_steps, warmup_steps=1000)
    
    try:
        net_g, optim_g, scheduler_g, _, epoch_str, step = utils.load_checkpoint(utils.latest_checkpoint_path(hps.train.save_dir, "G_*.pth"), net_g, optim_g, scheduler_g)
        net_d, optim_d, scheduler_d, _, epoch_str, step = utils.load_checkpoint(utils.latest_checkpoint_path(hps.train.save_dir, "D_*.pth"), net_d, optim_d, scheduler_d)
        global_step = step
    except:
        import traceback
        traceback.print_exc()
        net_g = utils.load_checkpoint_weight_only(utils.latest_checkpoint_path(hps.train.save_dir, "G_*.pth"), net_g)
        net_d = utils.load_checkpoint_weight_only(utils.latest_checkpoint_path(hps.train.save_dir, "D_*.pth"), net_d)
        epoch_str = 1
        global_step = 0

    train_loader = DataLoader(
        dataset,
        batch_size=hps.train.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=False
    )

    (
        net_g, 
        optim_g, 
        scheduler_g, 
        net_d, 
        optim_d, 
        scheduler_d, 
        train_loader,
        msspec,
        wavlm,
        wl_loss
    ) = accelerator.prepare(
        net_g, 
        optim_g, 
        scheduler_g, 
        net_d, 
        optim_d, 
        scheduler_d, 
        train_loader,
        msspec,
        wavlm,
        wl_loss
    )

    for epoch in range(epoch_str, hps.train.epochs + 1):
        dataset.refresh_epoch(epoch)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{hps.train.epochs}"):
            if batch == None:
                continue

            if accelerator.is_main_process:
                train_and_evaluate(
                    accelerator, 
                    epoch, 
                    hps, 
                    [net_g, net_d, msspec, wavlm, wl_loss], 
                    [optim_g, optim_d], 
                    [scheduler_g, scheduler_d], 
                    batch, 
                    logger, 
                    [writer, writer_eval]
                )
            else:
                train_and_evaluate(
                    accelerator, 
                    epoch, 
                    hps, 
                    [net_g, net_d, msspec, wavlm, wl_loss], 
                    [optim_g, optim_d], 
                    [scheduler_g, scheduler_d], 
                    batch, 
                    None, 
                    None
                )
            scheduler_g.step()
            scheduler_d.step()
            if global_step >= 100000000000000:
                logger.info('End training at step 100w')
                break
        


def train_and_evaluate(
        accelerator: accelerate.Accelerator, 
        epoch, 
        hps, 
        nets, 
        optims, 
        schedulers, 
        item, 
        logger, 
        writers
    ):
    net_g, net_d, msspec, wavlm, wl_loss = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    wavs, keys= item['wav'], item['utt']
    if writers is not None:
        writer, writer_eval = writers

    global global_step

    net_msstftd = net_d.module["net_msstftd"]
    net_wld = net_d.module["net_wld"]
    net_dacd = net_d.module["net_dacd"]
    

    wl_loss = wl_loss.module if hasattr(wl_loss, 'module') else wl_loss
    wavlm = wavlm.module if hasattr(wavlm, 'module') else wavlm
    dacd_dis_loss = net_dacd.module if hasattr(net_dacd, 'module') else net_dacd
    
    net_g.train()
    net_msstftd.train()
    net_wld.train()
    net_dacd.train()

    wavs = wavs.unsqueeze(1).cuda()
    q_semantic, res, semantic_loss, semantic_features , acoustic_guide_l1_loss = net_g(wavs)
    
    wavlm_semantic_loss = wavlm(wavs, semantic_features.transpose(1,2)) * 50
    # disc loss
    y_d_hat_r, y_d_hat_g, _, _ = net_msstftd(wavs, res.x.detach())
    
    # WLD
    if global_step > hps.train.start_wd_step:
        loss_disc_w = wl_loss.discriminator(
            wavs.squeeze(1), res.x.detach().squeeze(1)
        ).mean()
    else:
        loss_disc_w = torch.tensor(0)
        
    # DAC判别器
    if global_step > hps.train.start_dacd_step:
        loss_disc_dac = dacd_dis_loss.discriminator_loss(res.x.detach().to(torch.float32), wavs.to(torch.float32))
    else:
        loss_disc_dac = torch.tensor(0)
        
    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
    
    loss_disc_all = loss_disc + loss_disc_w + loss_disc_dac

    optim_d.zero_grad()
    accelerator.backward(loss_disc_all)
    
    # 梯度裁剪
    grad_norm_msstftd = clip_grad_value_(net_msstftd.parameters(), 1.0)
    grad_norm_wld = clip_grad_value_(net_wld.parameters(), 1.0)
    grad_norm_dacd = clip_grad_value_(net_dacd.parameters(), 1.0)
    
    optim_d.step()
    wavs = wavs.to(res.x.device)
    # loss
    y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_msstftd(wavs.detach(), res.x)
    loss_msspec = msspec(res.x, wavs) * 15
    
    # WLD loss
    if global_step > hps.train.start_wd_step:
        loss_fm_w = wl_loss(wavs.detach().squeeze(1),res.x.squeeze(1)).mean()
        loss_gen_w = wl_loss.generator(res.x.squeeze(1))
    else:
        loss_fm_w = torch.tensor(0)
        loss_gen_w = torch.tensor(0)
        
    if global_step > hps.train.start_dacd_step:
        # DAC loss
        dacd_gen_loss, dacd_fm_loss = dacd_dis_loss.generator_loss(res.x.to(torch.float32), wavs.detach().to(torch.float32))
    else:
        dacd_gen_loss = torch.tensor(0)    
        dacd_fm_loss = torch.tensor(0)  
    
    loss_fm = feature_loss(fmap_r, fmap_g)
    loss_gen, losses_gen = generator_loss(y_d_hat_g)
    loss_gen_all = 0.5 * acoustic_guide_l1_loss + wavlm_semantic_loss + loss_gen + loss_fm + loss_msspec + 50 * semantic_loss + 10*res.penalty + 10*q_semantic.loss + loss_fm_w + loss_gen_w + dacd_gen_loss + dacd_fm_loss
   
    optim_g.zero_grad()
    accelerator.backward(loss_gen_all)

    grad_norm_g = clip_grad_value_(net_g.parameters(), 1.0)
    optim_g.step()
    
    if accelerator.is_main_process:
        if global_step % hps.train.log_interval == 0:
            logger.info('====> Epoch: {}'.format(epoch))
            logger.info([
                global_step, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                loss_msspec.item(),
                acoustic_guide_l1_loss.item(),
                wavlm_semantic_loss.item(),
                50*semantic_loss.item(),
                loss_fm.item(),
                loss_gen.item(),
                loss_fm_w.item(),
                loss_gen_w.item(),
                dacd_fm_loss.item(),
                dacd_gen_loss.item(),
                res.penalty.item(),
                q_semantic.loss.item(),
                loss_disc.item(),
                loss_disc_w.item(),
                loss_disc_dac.item(),
            ])
            lr = optim_g.param_groups[0]['lr']
            
            scalar_dict = {
                "loss/total": loss_gen_all, 
                "loss/msspec": loss_msspec,
                "loss/acoustic_guide_l1_loss": acoustic_guide_l1_loss,
                "loss/wavlm_semantic_loss": wavlm_semantic_loss,
                "loss/semantic": 50*semantic_loss,
                "loss/commit_loss_se": q_semantic.loss,
                "loss/commit_loss_ac": res.penalty,
                "loss/stft_gen": loss_gen,
                "loss/stft_fm": loss_fm,
                "loss/stft_disc": loss_disc,
                "loss/wavlm_gen": loss_gen_w,
                "loss/wavlm_fm": loss_fm_w,
                "loss/wavlm_disc": loss_disc_w,
                "loss/dacd_gen": dacd_gen_loss,
                "loss/dacd_fm": dacd_fm_loss,
                "loss/dacd_disc": loss_disc_dac,
                "loss/disc": loss_disc_all,
                "learning_rate": lr,
                "grad_norm_g": grad_norm_g,
                "grad_norm_msstftd": grad_norm_msstftd,
                "grad_norm_wld": grad_norm_wld,
                "grad_norm_dacd": grad_norm_dacd
            }
            scalar_dict.update(res.metrics)  
            scalar_dict.update(q_semantic.metrics)
            
            utils.summarize(
                writer=writer,
                global_step=global_step, 
                scalars=scalar_dict
            )

        if global_step % hps.train.eval_interval == 0:
            logger.info(['All training params(G): ', utils.count_parameters(net_g), ' M'])

        if global_step % hps.train.eval_interval == 0:
            utils.save_checkpoint(net_g, optim_g, scheduler_g, hps.train.learning_rate, epoch, global_step, os.path.join(hps.train.save_dir, "G_{}.pth".format(global_step)))
            utils.save_checkpoint(net_d, optim_d, scheduler_d, hps.train.learning_rate, epoch, global_step, os.path.join(hps.train.save_dir, "D_{}.pth".format(global_step)))
            net_g.train()
    global_step += 1
    
 
if __name__ == "__main__":
    main()