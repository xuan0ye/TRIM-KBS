"""Shared trainer: fixed source split, paired batches/masks, global dev selection."""
import gc,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from transformers import get_cosine_schedule_with_warmup
from common import *
from models import make_model,configure_phase,optimizer
def save_torch(path,obj):
 path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def cpu_state(model):return {k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
def rng_state():
 return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()if torch.cuda.is_available()else None)
def restore_rng(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
 if s['cuda']is not None:torch.cuda.set_rng_state_all(s['cuda'])
def loader(d,batch,shuffle=False,generator=None):
 return DataLoader(TensorDataset(*(torch.as_tensor(d[k])for k in ['text','audio','vision','y','observed'])),batch_size=batch,shuffle=shuffle,num_workers=0,generator=generator)
def loss_function(y,device):
 counts=np.bincount(y,minlength=6)
 if np.any(counts==0):raise ValueError('All six training classes required')
 w=np.sqrt(counts.sum()/(6*counts));w/=w.mean()
 return torch.nn.CrossEntropyLoss(weight=torch.as_tensor(w,dtype=torch.float32,device=device),label_smoothing=plan()['label_smoothing'])
def evaluate_dev(model,d,device,batch=32):
 model.eval();scores=[]
 with torch.inference_mode():
  for t,a,v,y,observed in loader(d,batch):
   z,_=model(t.to(device),a.to(device),v.to(device),observed.to(device));scores.append(z.cpu().numpy())
 margin=np.concatenate(scores);return margin,metrics(d['y'],margin)
def signature(out,dataset,variant,seed):
 p=plan();meta=read(Path(out)/'data'/dataset/'prepared.json')
 return dict(dataset=dataset,variant=variant,seed=seed,protocol_sha256=sha(HERE/'protocol.json'),bundle_sha256=verify_code(),
  prepared_signature=meta['signature'],fit_sha256=meta['files']['fit.npz'],dev_sha256=meta['files']['dev.npz'],bert_identity_sha256=sha(Path(out)/'bert_identity.json'),
  phases=p['phases'][dataset],torch=torch.__version__,numpy=np.__version__,batch=p['batch_size'],accumulation=p['accumulation_steps'],test_evaluated=False)
def train_one(root,out,dataset,variant,seed,device):
 p=plan();out=Path(out);folder=out/'training'/dataset/variant/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
 sig=signature(out,dataset,variant,seed);complete=folder/'complete.json'
 if complete.exists():
  old=read(complete);assert old['signature']==sig,'Completed job signature changed'
  for n,h in old['files'].items():assert sha(folder/n)==h,n
  print('IEMO_TRAIN_CACHED',dataset,variant,seed,flush=True);return
 runtime(device,seed);tr=selection_data(out,dataset,'fit');va=selection_data(out,dataset,'dev')
 resume_path=folder/'resume.pt';resume=torch.load(resume_path,map_location='cpu',weights_only=False)if resume_path.exists()else None
 if resume and resume['signature']!=sig:raise RuntimeError('Resume signature differs; preserve the old run and use a new directory.')
 bert=Path(root)/'MSA Datasets/pretrained_berts'/('bert_cn'if dataset=='SIMS'else'bert_en')
 model=make_model(bert,tr['audio'].shape[1],tr['vision'].shape[1],variant,pretrained=resume is None).to(device)
 if resume:model.load_state_dict(resume['model'],strict=True)
 else:dump(folder/'model_identity.json',model.identity())
 identity=read(folder/'model_identity.json')
 g=torch.Generator().manual_seed(seed+11001);mg=torch.Generator().manual_seed(seed+22003)
 tl=loader(tr,p['batch_size'],True,g);criterion=loss_function(tr['y'],device)
 history=resume['history']if resume else [];best=resume['best']if resume else {'macro_f1':-1}
 # Separate stochastic model operations from the data-order and modality-mask RNGs.
 torch.manual_seed(seed+33007)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed+33007)
 started=time.monotonic()
 print('IEMO_TRAIN_START',dataset,variant,seed,'fit',len(tr['y']),'dev',len(va['y']),flush=True)
 for phase,epochs in enumerate(p['phases'][dataset]):
  if resume and phase<resume['phase']:continue
  configure_phase(model,phase);opt=optimizer(model,phase)
  steps=math.ceil(len(tl)/p['accumulation_steps']);total=steps*epochs
  sched=get_cosine_schedule_with_warmup(opt,max(1,int(total*.1)),total)
  first=1;phase_best=-1.;phase_file=folder/f'phase_{phase}_best.pt'
  if resume and phase==resume['phase']:
   opt.load_state_dict(resume['optimizer']);sched.load_state_dict(resume['scheduler']);restore_rng(resume['rng'])
   g.set_state(resume['loader_rng']);mg.set_state(resume['mask_rng']);first=resume['epoch']+1;phase_best=resume['phase_best']
  for epoch in range(first,epochs+1):
   tick=time.monotonic();model.train();opt.zero_grad(set_to_none=True);loss_sum=0.;mask_hash=hashlib.sha256()
   for k,(t,a,v,y,observed)in enumerate(tl):
    mask=observed.bool()&(torch.rand((len(y),2),generator=mg)>=p['train_modality_dropout']);mask_hash.update(mask.numpy().tobytes())
    t,a,v,y=t.to(device),a.to(device),v.to(device),y.long().to(device)
    z,aux=model(t,a,v,mask.to(device),auxiliary=True)
    loss=criterion(z,y)+p['auxiliary']['nce']*aux['nce']+p['auxiliary']['ot']*aux['ot']
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
    actual=min(p['accumulation_steps'],len(tl)-(k//p['accumulation_steps'])*p['accumulation_steps'])
    (loss/actual).backward();loss_sum+=float(loss.detach())
    if (k+1)%p['accumulation_steps']==0 or k+1==len(tl):
     torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();sched.step();opt.zero_grad(set_to_none=True)
   margin,val=evaluate_dev(model,va,device)
   if val['macro_f1']>phase_best:phase_best=val['macro_f1'];save_torch(phase_file,cpu_state(model))
   if val['macro_f1']>best['macro_f1']:
    best=dict(macro_f1=val['macro_f1'],phase=phase+1,epoch=epoch);save_torch(folder/'best.pt',cpu_state(model));dump(folder/'best.json',best)
   row=dict(phase=phase+1,epoch=epoch,loss=loss_sum/len(tl),dev_macro_f1=val['macro_f1'],dev_accuracy=val['accuracy'],modality_mask_sha256=mask_hash.hexdigest(),seconds=round(time.monotonic()-tick,2),loader_rng_sha256=hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest())
   history.append(row);csvout(folder/'history.csv',history)
   save_torch(resume_path,dict(signature=sig,phase=phase,epoch=epoch,phase_best=phase_best,best=best,history=history,model=cpu_state(model),
    optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state(),loader_rng=g.get_state(),mask_rng=mg.get_state()))
   dump(out/'current_job.json',dict(status='training',dataset=dataset,variant=variant,seed=seed,latest=row,best=best,test_evaluated=False))
   print('IEMO_EPOCH',dataset,variant,seed,'P',phase+1,'E',epoch,'dev_f1',round(val['macro_f1'],6),'seconds',row['seconds'],flush=True)
  model.load_state_dict(torch.load(phase_file,map_location='cpu',weights_only=True),strict=True)
  resume=None;del opt,sched;gc.collect()
 model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True),strict=True)
 margin,val=evaluate_dev(model,va,device);assert abs(val['macro_f1']-best['macro_f1'])<1e-10
 npz(folder/'dev_predictions.npz',ids=va['ids'],labels=va['y'],logits=margin)
 dump(folder/'dev_metrics.json',val)
 names=['best.pt','best.json','model_identity.json','history.csv','dev_predictions.npz','dev_metrics.json']
 dump(complete,dict(signature=sig,best=best,identity=identity,files={n:sha(folder/n)for n in names},seconds=time.monotonic()-started,test_evaluated=False))
 resume_path.unlink(missing_ok=True)
 for phase in range(len(p['phases'][dataset])):(folder/f'phase_{phase}_best.pt').unlink(missing_ok=True)
 print('IEMO_TRAIN_DONE',dataset,variant,seed,'best_dev_f1',best['macro_f1'],flush=True)
 del model;gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()
def seal_training(out):
 p=plan();files={}
 for dataset in p['datasets']:
  for variant in p['variants']:
   for seed in p['seeds']:
    folder=Path(out)/'training'/dataset/variant/f'seed_{seed}';m=read(folder/'complete.json')
    assert m['signature']['protocol_sha256']==sha(HERE/'protocol.json')and not m['test_evaluated']
    for name,digest in m['files'].items():assert sha(folder/name)==digest
    files[(folder/'best.pt').relative_to(out).as_posix()]=m['files']['best.pt']
 dump(Path(out)/'training_seal.json',dict(protocol_sha256=sha(HERE/'protocol.json'),bundle_sha256=verify_code(),jobs=len(files),checkpoint_sha256=files,
  selection='inner dev only; all checkpoints sealed before Session 5 evaluation',test_evaluated=False))