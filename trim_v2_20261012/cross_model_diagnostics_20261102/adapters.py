"""Frozen, native-input adapters. Each worker handles one architecture only."""
from pathlib import Path
from unittest.mock import patch
import gc,importlib,json,sys
import numpy as np
import torch
from torch import nn
from transformers import BertConfig,BertModel
from diag_common import HERE,sequence_order

class Adapter:
 def __init__(self,root,dataset,method,checkpoint,device):
  self.method=method;self.device=device;self.dataset=dataset;self.memo_hits=0;self.memo_calls=0;self.memo_enabled=True;self.temporal_changed={}
  sys.path.insert(0,str(HERE/'_trim'))
  import common as tc
  self.tc=tc
  if method=='TRIM':
   import model_tools as tm
   raw=tc.read_raw(root/'MSA Datasets',dataset);self.data=tc.prepare(raw['test']);del raw;gc.collect()
   self.ids=self.data['ids'];self.y=self.data['y'];self.t=self.data['text'];self.a=self.data['audio'];self.v=self.data['vision'];self.tm=tm
   bert=root/'MSA Datasets/pretrained_berts'/('bert_cn'if dataset=='SIMS'else'bert_en')
   self.model=tm.make_model(bert,self.a.shape[-1],self.v.shape[-1]).to(device)
  elif method=='MulT':
   sys.path.insert(0,str(HERE/'_mult'));import run_cormult_repro as cm
   path=cm.resolve_dataset_pkl(dataset,str(root/'MSA Datasets'))
   with path.open('rb')as f:raw=tc.mm.load(f)
   ds=cm.VWBSequenceDataset(raw['test']);del raw;gc.collect()
   self.ids=np.asarray([r[0]for r in ds.items]);self.y=np.asarray([r[4]for r in ds.items],dtype=np.int8)
   # Released processed arrays have fixed stored lengths. Refuse to silently change native variable-length collation.
   for col in [1,2,3]:
    if len({tuple(r[col].shape)for r in ds.items})!=1:raise RuntimeError('Variable native sequence shapes require explicit collation review')
   self.t=np.stack([r[1].numpy()for r in ds.items]);self.a=np.stack([r[2].numpy()for r in ds.items]);self.v=np.stack([r[3].numpy()for r in ds.items])
   self.model=cm.CorMulTReproModel(ds.text_dim,ds.audio_dim,ds.vision_dim,d=30,corr_d_model=128,corr_out_dim=64,
    num_heads=2,layers=2,dropout=.1,out_dropout=0.,use_correlation=False,corr_model=None,mlp_head=False).to(device)
  else:
   import yaml
   sys.path.insert(0,str(HERE/'_lnln'));from core.dataset import MMDataset
   from models.lnln import build_model
   args=yaml.safe_load((HERE/'_lnln/configs'/f'train_{dataset.lower()}.yaml').read_text('utf8'))
   path=root/'MSA Datasets'/dataset/'Processed'/('unaligned_39.pkl'if dataset=='SIMS'else'unaligned_50.pkl')
   bert=root/'MSA Datasets/pretrained_berts'/('bert_cn'if dataset=='SIMS'else'bert_en')
   args['dataset']['dataPath']=str(path);args['base']['missing_rate_eval_test']=0.;args['base']['num_workers']=0
   args['model']['feature_extractor']['bert_pretrained']=str(bert)
   with patch('pickle.load',side_effect=lambda f,*a,**kw:tc.mm.load(f)):ds=MMDataset(args,mode='test')
   labels=ds.labels['M'].reshape(-1);keep=np.flatnonzero(labels!=0)
   self.ids=np.asarray([str(ds.ids[i])for i in keep]);self.y=(labels[keep]>0).astype(np.int8)
   self.t=ds.text_m[keep].astype(np.float32);self.a=ds.audio_m[keep].astype(np.float32);self.v=ds.vision_m[keep].astype(np.float32)
   config=BertConfig.from_pretrained(str(bert),local_files_only=True)
   # The full task checkpoint supplies every BERT tensor; avoid loading a second full pretrained copy.
   with patch.object(BertModel,'from_pretrained',side_effect=lambda *a,**kw:BertModel(config)):
    self.model=build_model(args).to(device)
  state=torch.load(checkpoint,map_location='cpu',weights_only=True)
  self.model.load_state_dict(state,strict=True);del state;self.model.eval().requires_grad_(False)
  self.parameters=sum(p.numel()for p in self.model.parameters())
  self.score_type='regression_score'if method=='LNLN'else'positive_class_probability'
  self.temporal_note='structural_identity_after_pooled_inputs'if method=='TRIM'else'nonzero_row_proxy_permutation'
  if method=='LNLN':
   original=self.model.bertmodel.forward
   def memo(text):
    if not self.memo_enabled:return original(text)
    if self.memo_value is None:
     self.memo_text=text.clone();self.memo_value=original(text).clone();self.memo_calls+=1
    elif not torch.equal(text,self.memo_text):raise RuntimeError('Text encoder memo requested with changed text')
    else:self.memo_hits+=1
    return self.memo_value.clone()
   self.model.bertmodel.forward=memo
  if method=='TRIM':
   hs=[[],[],[]]
   with torch.inference_mode():
    for st in range(0,len(self.y),32):
     ix=slice(st,st+32);h=self.tm.encode(self.model,self.tensor(self.t[ix]),self.tensor(self.a[ix]),self.tensor(self.v[ix]))
     for j in range(3):hs[j].append(h[j].cpu())
   self.ht,self.ha,self.hv=[torch.cat(x)for x in hs]
   with torch.inference_mode():
    self.za=self.model.proj_a(torch.zeros((1,self.a.shape[-1]),device=device)).cpu()
    self.zv=self.model.proj_v(torch.zeros((1,self.v.shape[-1]),device=device)).cpu()
  for array in [self.a,self.v]:
   if not np.isfinite(array).all():raise RuntimeError('Nonfinite native model input')
 def tensor(self,a):return torch.as_tensor(a,device=self.device)
 def start(self,ix):
  self.ix=np.asarray(ix);self.memo_text=None;self.memo_value=None
  self.batch_text=self.tensor(self.t[self.ix])
  if self.method=='TRIM':
   self.bht=self.ht[self.ix].to(self.device)
  return self
 def inputs(self,case,donors):
  ix=self.ix;kind=case['kind'];ia=iv=ix
  if kind=='audio_cross_video':ia=donors[case['name']][0][ix]
  if kind=='vision_cross_video':iv=donors[case['name']][1][ix]
  if kind=='paired_av_cross_video':ia=iv=donors[case['name']][0][ix]
  if self.method=='TRIM':
   a=self.ha[ia].to(self.device);v=self.hv[iv].to(self.device)
   if kind in ['zero_audio_present','zero_av_present']:a=self.za.to(self.device).expand(len(ix),-1)
   if kind in ['zero_vision_present','zero_av_present']:v=self.zv.to(self.device).expand(len(ix),-1)
   return a,v
  a=self.a[ia];v=self.v[iv]
  if kind in ['audio_order','av_order']:
   a,_=sequence_order(a,self.ids[ix],20261102+case['rep']*101)
   self.temporal_changed.setdefault(case['name'],set()).update(ix[np.any(a!=self.a[ix],axis=(1,2))].tolist())
  if kind in ['vision_order','av_order']:
   v,_=sequence_order(v,self.ids[ix],20261103+case['rep']*101)
   self.temporal_changed.setdefault(case['name'],set()).update(ix[np.any(v!=self.v[ix],axis=(1,2))].tolist())
  a=self.tensor(a);v=self.tensor(v)
  if kind in ['zero_audio_present','zero_av_present']:a=torch.zeros_like(a)
  if kind in ['zero_vision_present','zero_av_present']:v=torch.zeros_like(v)
  return a,v
 def predict(self,case,donors):
  a,v=self.inputs(case,donors);mask=case['mask']
  if self.method=='TRIM':
   m=torch.tensor(mask,device=self.device,dtype=torch.float32)[None].expand(len(self.ix),-1)
   output=self.tm.fuse(self.model,self.bht,a,v,m)
  else:
   if not mask[1]:a=torch.zeros_like(a)
   if not mask[2]:v=torch.zeros_like(v)
   if self.method=='MulT':output=self.model(self.batch_text,a,v)
   else:output=self.model((None,None,None),(v,a,self.batch_text))['sentiment_preds']
  if not torch.isfinite(output).all():raise RuntimeError('Nonfinite model output')
  if self.method=='LNLN':score=output.reshape(-1);pred=score>0
  else:score=output.float().softmax(-1)[:,1];pred=output.argmax(-1)
  return pred.to(torch.int8).cpu().numpy(),score.float().cpu().numpy(),output.detach().cpu()
 def check_native(self,definitions,donors):
  errors={}
  if self.method=='TRIM':
   with torch.inference_mode():
    native,_=self.model(self.batch_text[:,0],self.batch_text[:,1],self.batch_text[:,2],self.tensor(self.a[self.ix]),self.tensor(self.v[self.ix]),training=False)
    got=self.predict(definitions[0],donors)[2]
   diff=float((native.cpu()-got).abs().max());assert diff<1e-5,diff;errors['pooled_wrapper_max_logit_error']=diff
  elif self.method=='LNLN':
   # Check memoized versus ordinary complete network for masks and actual donor/order changes on this real batch.
   check=[c for c in definitions if c['name'] in ['full','text_only','text_audio','text_visual','audio_cross_video_00','audio_order_00']]
   for case in check:
    self.memo_enabled=False;plain=self.predict(case,donors)[2];self.memo_enabled=True;self.memo_text=None;self.memo_value=None
    cached=self.predict(case,donors)[2];difference=float((plain-cached).abs().max());assert difference<1e-6,difference
    # A second call must actually hit the memo rather than only warming it.
    repeat=self.predict(case,donors)[2];assert float((cached-repeat).abs().max())<1e-6
    errors[case['name']]=difference
   self.memo_text=None;self.memo_value=None
  return errors