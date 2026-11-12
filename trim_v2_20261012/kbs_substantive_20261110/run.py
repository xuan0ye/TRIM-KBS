"""Orchestrator. Full run: verify, prepare, smoke, train 75, seal, evaluate, summarize."""
import argparse,contextlib,gc,os,platform,shutil,sys,time
from pathlib import Path
import numpy as np
from common import *
def dependencies(device,require_space=False):
 import torch,transformers
 if not torch.__version__.startswith('2.4.'):raise RuntimeError('Expected tested PyTorch 2.4.x; got '+torch.__version__)
 if transformers.__version__!='4.49.0':raise RuntimeError('Expected Transformers 4.49.0; got '+transformers.__version__)
 if not np.__version__.startswith('1.26.'):raise RuntimeError('Expected NumPy 1.26.x; got '+np.__version__)
 if str(device).startswith('cuda')and not torch.cuda.is_available():raise RuntimeError('No CUDA: activate wjhenv and set CUDA_VISIBLE_DEVICES=0.')
 return dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,cuda=torch.version.cuda,
  device=str(device),gpu=torch.cuda.get_device_name(0)if torch.cuda.is_available()else None)
@contextlib.contextmanager
def global_lock():
 path=HERE.parent/'logs/suite.lock';path.parent.mkdir(exist_ok=True)
 with path.open('a+')as handle:
  if os.name!='nt':
   import fcntl
   try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:raise RuntimeError('Another remote experiment holds ../logs/suite.lock. Existing jobs were not stopped.')
  yield
def preflight(root,out,device,quick=False):
 code=verify_code();rt=dependencies(device)
 for dataset,rel in REL.items():
  if not (Path(root)/'MSA Datasets'/rel).is_file():raise FileNotFoundError(Path(root)/'MSA Datasets'/rel)
 for lang in ['bert_en','bert_cn']:
  if not (Path(root)/'MSA Datasets/pretrained_berts'/lang/'config.json').is_file():raise FileNotFoundError(lang)
 if quick:print('KBS_QUICK_CHECK_OK',rt,flush=True);return
 from data import verify_berts
 identity=verify_berts(root,out)
 dump(Path(out)/'runtime.json',dict(runtime=rt,bundle_sha256=code,protocol_sha256=sha(HERE/'protocol.json')))
def disk_check(out):
 p=plan();jobs=len(p['datasets'])*len(p['variants'])*len(p['seeds'])
 # Five model families have around 110--125M FP32 parameters each; reserve
 # one full checkpoint per job plus current optimizer/phase caches.
 required=jobs*125_000_000*4+5_000_000_000
 free=shutil.disk_usage(Path(out)).free
 if free<required:raise RuntimeError(f'Insufficient checkpoint space: free={free/1e9:.1f} GB, conservative requirement={required/1e9:.1f} GB')
 print('KBS_DISK_OK','free_GB',round(free/1e9,1),'reserved_GB',round(required/1e9,1),flush=True)
def smoke(root,out,device):
 import torch
 from models import make_model,configure_phase,optimizer,value_only_block
 from train import loss_function
 p=plan();signature=dict(protocol_sha256=sha(HERE/'protocol.json'),bundle_sha256=verify_code(),bert_identity_sha256=sha(Path(out)/'bert_identity.json'))
 existing=Path(out)/'smoke_check.json'
 if existing.exists()and read(existing)['signature']==signature:
  print('KBS_SMOKE_CACHED',flush=True);return
 records=[]
 for dataset in p['datasets']:
  tr=selection_data(out,dataset,'fit');ix=np.r_[np.flatnonzero(tr['y']==0)[:4],np.flatnonzero(tr['y']==1)[:4]]
  t=torch.as_tensor(tr['text'][ix],device=device);a=torch.as_tensor(tr['audio'][ix],device=device);v=torch.as_tensor(tr['vision'][ix],device=device);y=torch.as_tensor(tr['y'][ix],dtype=torch.long,device=device)
  bert=Path(root)/'MSA Datasets/pretrained_berts'/('bert_cn'if dataset=='SIMS'else'bert_en')
  for variant in p['variants']:
   runtime(device,101);model=make_model(bert,a.shape[1],v.shape[1],variant).to(device);configure_phase(model,0)
   criterion=loss_function(tr['y'],device);opt=optimizer(model,0);model.train()
   z,aux=model(t,a,v,torch.ones((len(ix),2),dtype=torch.bool,device=device),auxiliary=True)
   loss=criterion(z,y)+.02*aux['nce']+.02*aux['ot'];assert torch.isfinite(loss);loss.backward()
   for n,value in model.named_parameters():
    if value.grad is not None:assert torch.isfinite(value.grad).all(),n
   torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();model.eval()
   with torch.inference_mode():
    off=torch.zeros((len(ix),2),dtype=torch.bool,device=device)
    x,_=model(t,a,v,off);xx,_=model(t,a*37+10,v*51-2,off)
    error=float((x-xx).abs().max());assert error<1e-5
   records.append(dict(dataset=dataset,variant=variant,finite_train_step=True,absent_input_max_delta=error,identity=model.identity()))
   print('KBS_SMOKE_OK',dataset,variant,flush=True);del model,opt;gc.collect()
   if torch.cuda.is_available():torch.cuda.empty_cache()
 dump(existing,dict(signature=signature,status='passed',records=records,formal_training_jobs=0,official_valid_test_evaluated=False))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['check','prepare','smoke','train','evaluate','summarize','run'])
 ap.add_argument('--root',type=Path,default=HERE.parent.parent);ap.add_argument('--out',type=Path,default=HERE/'runs/substantive')
 ap.add_argument('--device',default='cuda:0');ap.add_argument('--quick',action='store_true')
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if a.mode=='check':preflight(a.root,a.out,a.device,a.quick);return
 if a.mode=='summarize':
  from summarize import summarize
  summarize(a.out);return
 with global_lock():
  if a.mode in ['prepare','smoke','train','run']:
   preflight(a.root,a.out,a.device)
   from data import prepare
   prepare(a.root,a.out)
   if a.mode=='prepare':return
   smoke(a.root,a.out,a.device)
   if a.mode=='smoke':return
  if a.mode in ['train','run']:
   disk_check(a.out)
   from train import train_one,seal_training
   p=plan();total=len(p['datasets'])*len(p['variants'])*len(p['seeds']);done=0
   for dataset in p['datasets']:
    for seed in p['seeds']:
     for variant in p['variants']:
      train_one(a.root,a.out,dataset,variant,seed,a.device);done+=1
      dump(a.out/'suite_progress.json',dict(stage='training',done=done,total=total,official_valid_test_evaluated=False))
   seal_training(a.out)
   if a.mode=='train':return
  if a.mode in ['evaluate','run']:
   from train import seal_training
   seal_training(a.out)
   from evaluate import evaluate_one
   p=plan();done=0;total=len(p['datasets'])*len(p['variants'])*len(p['seeds'])*len(p['evaluation_roles'])
   for dataset in p['datasets']:
    for variant in p['variants']:
     for seed in p['seeds']:
      for role in p['evaluation_roles']:
       evaluate_one(a.root,a.out,dataset,variant,seed,role,a.device);done+=1
       dump(a.out/'suite_progress.json',dict(stage='post_seal_evaluation',done=done,total=total,official_valid_test_evaluated=True,selection_performed=False))
   from summarize import summarize
   archive=summarize(a.out);dump(a.out/'suite_progress.json',dict(stage='complete',done=total,total=total,review_zip=str(archive),selection_performed=False))
if __name__=='__main__':main()