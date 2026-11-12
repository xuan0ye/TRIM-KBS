"""Reproducibility and staged optimization, with cuDNN explicitly disabled."""
import random,math,os
import numpy as np
import torch
def runtime(device="auto",threads=4):
 torch.backends.cudnn.enabled=False;torch.backends.cudnn.benchmark=False
 if hasattr(torch.backends.cuda,"enable_cudnn_sdp"):torch.backends.cuda.enable_cudnn_sdp(False)
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.set_num_threads(threads);torch.set_default_dtype(torch.float32)
 return torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if device=="auto" else device)
def seed_everything(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def rng_state(generator):
 return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,loader=generator.get_state())
def restore_rng(r,generator):
 random.setstate(r["python"]);np.random.set_state(r["numpy"]);torch.set_rng_state(r["torch"]);generator.set_state(r["loader"])
 if r["cuda"] is not None:torch.cuda.set_rng_state_all(r["cuda"])
def configure_phase(model,phase):
 for p in model.bert.parameters():p.requires_grad_(False)
 if phase==1:
  for layer in model.bert.encoder.layer[-2:]:
   for p in layer.parameters():p.requires_grad_(True)
 elif phase==2:
  for p in model.bert.parameters():p.requires_grad_(True)
def optimizer_for(model,phase):
 groups=[]
 b=[p for n,p in model.named_parameters() if n.startswith("bert.") and p.requires_grad]
 feature_prefix=("text_project.","audio.","visual.")
 feat=[p for n,p in model.named_parameters() if n.startswith(feature_prefix) and p.requires_grad]
 heads=[p for n,p in model.named_parameters() if not n.startswith(("bert.",)+feature_prefix) and p.requires_grad]
 if b:groups.append(dict(params=b,lr=2e-5 if phase==1 else 1e-5))
 groups.append(dict(params=feat,lr=3e-4 if phase==0 else 5e-5));groups.append(dict(params=heads,lr=3e-4))
 return torch.optim.AdamW(groups,weight_decay=.01)
def cpu_state(model):return {k:v.detach().cpu() for k,v in model.state_dict().items()}
def save_torch(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(".tmp");torch.save(value,temp);os.replace(temp,path)
