"""Evaluate sealed checkpoints only; never tune from official valid/test results."""
import gc,time,hashlib
from pathlib import Path
import numpy as np
import torch
from common import *
from models import make_model
from interventions import *
def encode_bank(model,data,device,batch=32):
 model.eval();ts=[];aa=[];vv=[]
 with torch.inference_mode():
  for start in range(0,len(data['y']),batch):
   ix=slice(start,start+batch);t=torch.as_tensor(data['text'][ix],device=device);a=torch.as_tensor(data['audio'][ix],device=device);v=torch.as_tensor(data['vision'][ix],device=device)
   ht=model.encode_text(t);ha,hv=model.encode_av(a,v);ts.append(ht.cpu());aa.append(ha.cpu());vv.append(hv.cpu())
  missing=model.absent_states(1,device);za,zv=model.encode_av(torch.zeros((1,data['audio'].shape[1]),device=device),torch.zeros((1,data['vision'].shape[1]),device=device))
 return dict(t=torch.cat(ts),a=torch.cat(aa),v=torch.cat(vv),ma=missing[0].cpu(),mv=missing[1].cpu(),za=za.cpu(),zv=zv.cpu())
def predictor(model,bank,device):
 def predict(ti,avi,mode='full'):
  ti=np.asarray(ti);avi=np.asarray(avi);t=bank['t'][ti].to(device)
  a=bank['a'][avi]if mode in ['full','text_audio']else bank['ma'].expand(len(ti),-1)
  v=bank['v'][avi]if mode in ['full','text_visual']else bank['mv'].expand(len(ti),-1)
  if mode=='zero_av_present':a=bank['za'].expand(len(ti),-1);v=bank['zv'].expand(len(ti),-1)
  with torch.inference_mode():z=model.fuse(t,a.to(device),v.to(device))
  return (z[:,1]-z[:,0]).float().cpu().numpy()
 return predict
def infer_batched(fn,i,j,mode='full',batch=256):
 return np.concatenate([fn(i[s:s+batch],j[s:s+batch],mode)for s in range(0,len(i),batch)])if len(i)else np.empty(0,np.float32)
def get_maps(out,dataset,role,data):
 p=Path(out)/'interventions'/dataset;p.mkdir(parents=True,exist_ok=True);path=p/f'{role}_maps.npz'
 seed=20261110+int.from_bytes(hashlib.sha256((dataset+'|'+role).encode()).digest()[:4],'little')
 edges=read(Path(out)/'data'/dataset/'prepared.json')['length_bin_edges'];maps=build_maps(data,edges,plan()['interventions']['repeats'],seed)
 if path.exists():
  old=arrays(path);assert set(old)==set(maps)
  for name in maps:assert np.array_equal(old[name],maps[name]),name
 else:npz(path,**maps)
 return maps,sha(path),seed
def evaluate_one(root,out,dataset,variant,seed,role,device):
 out=Path(out);data=evaluation_data(out,dataset,role);maps,map_hash,map_seed=get_maps(out,dataset,role,data)
 trainfolder=out/'training'/dataset/variant/f'seed_{seed}';weights=trainfolder/'best.pt';sealed=read(out/'training_seal.json')
 expected=sealed['checkpoint_sha256'][weights.relative_to(out).as_posix()]
 if sha(weights)!=expected:raise RuntimeError('Sealed checkpoint changed')
 folder=out/'evaluation'/dataset/variant/f'seed_{seed}'/role;folder.mkdir(parents=True,exist_ok=True)
 sig=dict(evaluation_data_sha256=sha(out/'data'/dataset/f'{role}.npz'),checkpoint_sha256=expected,training_seal_sha256=sha(out/'training_seal.json'),maps_sha256=map_hash,protocol_sha256=sha(HERE/'protocol.json'),bundle_sha256=verify_code())
 if (folder/'complete.json').exists():
  old=read(folder/'complete.json');assert old['signature']==sig
  for name,h in old['files'].items():assert sha(folder/name)==h
  print('KBS_EVALUATION_CACHED',dataset,variant,seed,role,flush=True);return
 runtime(device,20261110);bert=Path(root)/'MSA Datasets/pretrained_berts'/('bert_cn'if dataset=='SIMS'else'bert_en')
 model=make_model(bert,data['audio'].shape[1],data['vision'].shape[1],variant,pretrained=False).to(device)
 model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True);model.eval()
 started=time.monotonic();print('KBS_EVALUATION_START',dataset,variant,seed,role,'n',len(data['y']),flush=True)
 bank=encode_bank(model,data,device);fn=predictor(model,bank,device);ix=np.arange(len(data['y']))
 with torch.inference_mode():
  q=ix[:min(8,len(ix))];z,_=model(torch.as_tensor(data['text'][q],device=device),torch.as_tensor(data['audio'][q],device=device),torch.as_tensor(data['vision'][q],device=device))
  delta=float(np.max(np.abs((z[:,1]-z[:,0]).cpu().numpy()-fn(q,q))))
 if delta>1e-4:raise RuntimeError('Cached-branch native forward mismatch: '+str(delta))
 margins=[];eligible=[];names=[]
 for name in ['full','text_only','text_audio','text_visual','zero_av_present']:
  names.append(name);margins.append(infer_batched(fn,ix,ix,name));eligible.append(np.ones(len(ix),bool))
 for name,donor in maps.items():
  keep=donor>=0;value=np.full(len(ix),np.nan,np.float32)
  value[keep]=infer_batched(fn,ix[keep],donor[keep]);names.append(name);margins.append(value);eligible.append(keep)
  if name.endswith('_19'):print('KBS_INTERVENTIONS',dataset,variant,seed,role,name,flush=True)
 if dataset in plan()['emap']['exact_datasets']:
  em,emapmeta=exact_emap(len(ix),lambda a,b:fn(a,b));bank_values=np.asarray([em]);half_values=np.asarray([em])
 else:
  bank_values=[];half_values=[];details=[]
  for anchor in anchors_for(data['ids'],plan()['emap']['anchors'],plan()['emap']['banks'],map_seed+7):
   value,half,meta=anchor_emap(len(ix),anchor,lambda a,b:fn(a,b));bank_values.append(value);half_values.append(half);details.append(meta)
  bank_values=np.asarray(bank_values);half_values=np.asarray(half_values);em=bank_values.mean(0);emapmeta=dict(kind='MC_EMAP',banks=details,primary='mean of two 64-anchor projections; not two training seeds',bank_margin_rmse=float(np.sqrt(np.square(bank_values[0]-bank_values[1]).mean())))
 if variant in ['bert_mlp','additive']:
  error=float(np.max(np.abs(em-margins[0])))
  if error>1e-4:raise RuntimeError('EMAP changed a structurally additive model: '+str(error))
  emapmeta['additive_max_margin_error']=error
 names.append('emap');margins.append(em);eligible.append(np.ones(len(ix),bool))
 margin=np.stack(margins);valid=np.stack(eligible);rows=[]
 from data import subsets
 for sub,mask in subsets(data).items():
  for k,name in enumerate(names):
   keep=mask&valid[k];m=metrics(data['y'][keep],margin[k,keep]);base=(margin[0,keep]>0);changed=(margin[k,keep]>0)
   rows.append(dict(dataset=dataset,variant=variant,seed=seed,split=role,subset=sub,condition=name,**m,
    changed_fraction=float((base!=changed).mean())if keep.any()else None,
    corrected=int(((base!=data['y'][keep])&(changed==data['y'][keep])).sum()),
    broken=int(((base==data['y'][keep])&(changed!=data['y'][keep])).sum())))
 npz(folder/'predictions.npz',ids=data['ids'],labels=data['y'],condition_names=np.asarray(names),margin=margin,eligible=valid,emap_bank_margins=bank_values,emap_half_anchor_margins=half_values)
 csvout(folder/'metrics.csv',rows);dump(folder/'emap.json',emapmeta)
 files=['predictions.npz','metrics.csv','emap.json']
 dump(folder/'complete.json',dict(signature=sig,files={n:sha(folder/n)for n in files},cached_branch_max_margin_error=delta,seconds=time.monotonic()-started,
  evaluation_role=role,test_evaluated=(role=='test'),previously_examined_benchmark=True,selection_performed=False,checkpoint_training_seed=seed))
 print('KBS_EVALUATION_DONE',dataset,variant,seed,role,flush=True);del model,bank;gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()