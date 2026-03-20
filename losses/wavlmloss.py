import torch
import torchaudio
from transformers import AutoModel

def d_axis_distill_loss(feature, target_feature):
    n = min(feature.size(1), target_feature.size(1))
    distill_loss = - torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=1))).mean()
    return distill_loss

def t_axis_distill_loss(feature, target_feature, lambda_sim=1):
    n = min(feature.size(1), target_feature.size(1))
    l1_loss = torch.functional.l1_loss(feature[:, :n], target_feature[:, :n], reduction='mean')
    sim_loss = - torch.log(torch.sigmoid(torch.nn.functional.cosine_similarity(feature[:, :n], target_feature[:, :n], axis=-1))).mean()
    distill_loss = l1_loss + lambda_sim * sim_loss
    return distill_loss 


class WavLMLoss(torch.nn.Module):
    def __init__(self, ckpt_path, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(ckpt_path, use_safetensors=False)
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)
        self.avg_pooling = torch.nn.AvgPool1d(8, 4, 3)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def forward(self, wav, q):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16.squeeze(1)
            ).last_hidden_state
            wav_embeddings = self.avg_pooling(wav_embeddings.transpose(1, 2)).transpose(1, 2)

        # channel dim
        floss = d_axis_distill_loss(q, wav_embeddings)
        
        return floss
        

