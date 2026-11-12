"""Shared-mask evaluation and explicit audio/visual content interventions."""
from pathlib import Path
import json
import numpy as np
import torch
from common import *
from aggregate_core import scores,METRICS,MASK_ORDER,KEYS
from data_v2 import make_loader,to_device
def fixed_predictions(model,ds,batch,device,cache_representations=False):
 model.eval();pred={k:[] for k in MASK_ORDER};prob={k:[] for k in MASK_ORDER};saved={}
 with torch.inference_mode():
  for data in make_loader(ds,batch):
   data=to_device(data,device);reps=model.representations(data);c=model.components_from_repr(reps);logits=model.all_patterns(c)
   assert torch.isfinite(logits).all()
   for i,key in enumerate(MASK_ORDER):
    pred[key].extend(logits[:,i].argmax(-1).cpu().tolist());prob[key].extend(logits[:,i].softmax(-1)[:,1].cpu().tolist())
   if cache_representations:
    for key,value in reps.items():saved.setdefault(key,[]).append(value.cpu())
 return {k:np.array(v,dtype=np.int8) for k,v in pred.items()},{k:np.array(v,dtype=np.float64) for k,v in prob.items()},{k:torch.cat(v) for k,v in saved.items()}
def validation(model,ds,batch,device):
 pred,prob,_=fixed_predictions(model,ds,batch,device)
 results={k:scores(ds.y,pred[k],prob[k]) for k in MASK_ORDER}
 return dict(selection_score=float(np.mean([results[k]["macro_f1"] for k in MASK_ORDER])),complete_macro_f1=results["full"]["macro_f1"],text_macro_f1=results["text_only"]["macro_f1"],conditions=results)
def interventions(model,reps,batch,device):
 n=len(reps["text"]);rng=np.random.default_rng(20261012);pa=rng.permutation(n);pv=rng.permutation(n)
 arrays=dict(shuffle_audio_indices=pa,shuffle_visual_indices=pv)
 with torch.inference_mode():
  for name,shuffle_a,shuffle_v in (("shuffle_audio",True,False),("shuffle_visual",False,True),("shuffle_both",True,True)):
   pp=[];pr=[]
   for start in range(0,n,batch):
    sl=slice(start,start+batch);data={}
    for key,value in reps.items():
     if shuffle_a and key in ("audio","a_present"):data[key]=value[pa[sl]].to(device)
     elif shuffle_v and key in ("visual","v_present"):data[key]=value[pv[sl]].to(device)
     else:data[key]=value[sl].to(device)
    c=model.components_from_repr(data);logits=model.compose(c,torch.ones((len(data["text"]),2),device=device))
    pp.extend(logits.argmax(-1).cpu().tolist());pr.extend(logits.softmax(-1)[:,1].cpu().tolist())
   arrays[name]=np.array(pp,dtype=np.int8);arrays["prob_"+name]=np.array(pr,dtype=np.float64)
 return arrays
def seed_metrics(ds,pred,prob,**prefix):
 y=ds.y;rows=[];masks=[];pp=np.stack([pred[k] for k in MASK_ORDER]);pr=np.stack([prob[k] for k in MASK_ORDER]);ii=np.arange(len(y))
 for c,k in KEYS.items():rows.append(dict(**prefix,condition=c,**scores(y,pred[k],prob[k])))
 for p in RATES:
  rr=[]
  for ms in MASK_SEEDS:
   ix=draw_masks(len(y),p,ms);sc=scores(y,pp[ix,ii],pr[ix,ii]);rr.append(sc);masks.append(dict(**prefix,condition=f"p={p:.1f}",mask_seed=ms,**sc))
  rows.append(dict(**prefix,condition=f"p={p:.1f}",n_test=len(y),**{k:float(np.mean([r[k] for r in rr])) for k in METRICS}))
 return rows,masks
