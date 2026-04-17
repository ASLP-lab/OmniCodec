import os
import torch
import librosa
from tqdm import tqdm
import soundfile as sf
from utils import utils
from models.compression_omnicodec import build_model

device = "cuda" if torch.cuda.is_available() else "cpu"
hps = utils.get_hparams()
net_g = build_model(
       hps.model.sample_rate, 
       hps.model.frame_rate,
       hps.model.q_dimension,
       hps.model.semantic_dimension,
       hps.model.seanet,
       hps.model.transformer,
       hps.model.semantic_quantizer,
       hps.model.acoustic_quantizer
).to(device)
exp_name = "omnicodec"
ckpt = torch.load(f'pretrained_model/omnicodec.pth',map_location='cpu')
net_g.load_state_dict(ckpt,strict=False)
net_g.eval()

testset_path = f"./testset/speech"
out_path  = f"./outputs"
os.makedirs(out_path,exist_ok=True)
for it in tqdm(os.listdir(testset_path)):
       try:
              wav_path = os.path.join(testset_path,it.strip())
              output_path = os.path.join(out_path, it.strip())
              wav ,sr = librosa.load(wav_path,sr=24000)
              wav = torch.from_numpy(wav)
              net_g.to(device)
              wav = wav.to(device)
              with torch.no_grad():
                     _, copysyn_wav, _, _, _ = net_g(wav.unsqueeze(0).unsqueeze(0)) # q_semantic, q_acoustic, semantic_loss, semantic_features, acoustic_guide_l1_loss
              
              sf.write(out_path,copysyn_wav.x.squeeze().detach().cpu().numpy(),24000)
       except:
              import traceback
              print(traceback.print_exc())
              continue
