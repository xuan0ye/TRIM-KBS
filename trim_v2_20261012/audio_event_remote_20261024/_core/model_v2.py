"""TRIM-v2 research candidate: temporal evidence, explicit subset masks, residual corrections."""
from pathlib import Path
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel,BertConfig
VARIANTS=("v2","text","mean","no_uni","no_consistency","biased_subsets")
PATTERN_BITS=((0,0),(1,0),(0,1),(1,1))
def positions(n,d,device,dtype):
 p=torch.arange(n,device=device,dtype=torch.float32)[:,None]
 scale=torch.exp(torch.arange(0,d,2,device=device,dtype=torch.float32)*(-math.log(10000.)/d))
 out=torch.zeros(n,d,device=device);out[:,0::2]=torch.sin(p*scale);out[:,1::2]=torch.cos(p*scale)
 return out.to(dtype)
class TemporalEvidence(nn.Module):
 def __init__(self,raw_dim,d=128,k=8,heads=4,drop=.1):
  super().__init__();self.d=d;self.k=k
  self.register_buffer("feature_mean",torch.zeros(raw_dim));self.register_buffer("feature_std",torch.ones(raw_dim))
  self.project=nn.Sequential(nn.Linear(raw_dim,d),nn.LayerNorm(d),nn.GELU())
  # Neighbour mixing uses Linear, never cuDNN convolution.
  self.temporal=nn.Sequential(nn.Linear(3*d,d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,d))
  self.temporal_norm=nn.LayerNorm(d)
  self.queries=nn.Parameter(torch.randn(k,d)*.02)
  self.pool=nn.MultiheadAttention(d,heads,dropout=drop,batch_first=True)
  self.pool_norm=nn.LayerNorm(d)
  self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(drop),nn.Linear(2*d,d))
  self.out_norm=nn.LayerNorm(d);self.drop=nn.Dropout(drop)
 def forward(self,raw,valid,mean_only=False):
  valid=valid.bool();present=valid.any(1);x=((raw-self.feature_mean)/self.feature_std).clamp(-10,10)
  x=torch.where(valid[:,:,None],x,torch.zeros_like(x))
  if mean_only:
   x=x.sum(1,keepdim=True)/valid.sum(1).clamp_min(1)[:,None,None];h=self.project(x)
   safe=torch.ones((len(raw),1),dtype=torch.bool,device=raw.device)
  else:
   h=self.project(x)+positions(x.shape[1],self.d,x.device,x.dtype)[None]
   h=h*valid[:,:,None]
   previous=F.pad(h[:,:-1],(0,0,1,0));following=F.pad(h[:,1:],(0,0,0,1))
   h=self.temporal_norm(h+self.drop(self.temporal(torch.cat((previous,h,following),dim=-1))))*valid[:,:,None]
   safe=valid.clone();safe[~present,0]=True
  q=self.queries[None].expand(len(raw),-1,-1)
  attended,_=self.pool(q,h,h,key_padding_mask=~safe,need_weights=False)
  tokens=self.pool_norm(q+self.drop(attended));tokens=self.out_norm(tokens+self.drop(self.ff(tokens)))
  return tokens*present[:,None,None],present
class TRIMV2(nn.Module):
 def __init__(self,bert_path,a_dim,v_dim,variant="v2",d=128,k=8,heads=4,pretrained=True):
  super().__init__();assert variant in VARIANTS;self.variant=variant;self.d=d
  if pretrained:self.bert=BertModel.from_pretrained(str(bert_path),local_files_only=True,attn_implementation="sdpa")
  else:
   config=BertConfig.from_pretrained(str(bert_path),local_files_only=True);config._attn_implementation="sdpa";self.bert=BertModel(config)
  self.text_project=nn.Sequential(nn.Linear(self.bert.config.hidden_size,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(.1))
  self.audio=TemporalEvidence(a_dim,d,k,heads);self.visual=TemporalEvidence(v_dim,d,k,heads)
  self.text_head=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Dropout(.1),nn.Linear(d,2))
  self.audio_head=nn.Linear(d,2);self.visual_head=nn.Linear(d,2)
  self.cross_audio=nn.MultiheadAttention(d,heads,dropout=.1,batch_first=True)
  self.cross_visual=nn.MultiheadAttention(d,heads,dropout=.1,batch_first=True)
  self.audio_norm=nn.LayerNorm(d);self.visual_norm=nn.LayerNorm(d)
  self.audio_delta=self.residual_head(3*d,d);self.visual_delta=self.residual_head(3*d,d);self.joint_delta=self.residual_head(4*d,d)
  self.audio_gate=nn.Linear(2*d,1);self.visual_gate=nn.Linear(2*d,1);self.joint_gate=nn.Linear(3*d,1)
  for gate in (self.audio_gate,self.visual_gate,self.joint_gate):nn.init.zeros_(gate.weight);nn.init.zeros_(gate.bias)
  for head in (self.audio_delta,self.visual_delta,self.joint_delta):nn.init.normal_(head[-1].weight,std=.01);nn.init.zeros_(head[-1].bias)
 @staticmethod
 def residual_head(input_dim,d):
  return nn.Sequential(nn.Linear(input_dim,d),nn.GELU(),nn.Dropout(.1),nn.Linear(d,2))
 def set_feature_stats(self,stats):
  for name,encoder in (("audio",self.audio),("vision",self.visual)):
   encoder.feature_mean.copy_(torch.tensor(stats[name]["mean"],dtype=encoder.feature_mean.dtype))
   encoder.feature_std.copy_(torch.tensor(stats[name]["std"],dtype=encoder.feature_std.dtype))
 def representations(self,batch):
  tb=batch["text"];out=self.bert(input_ids=tb[:,0],attention_mask=tb[:,1],token_type_ids=tb[:,2],return_dict=True)
  t=self.text_project(out.last_hidden_state[:,0])
  if self.variant=="text":return dict(text=t)
  aa,pa=self.audio(batch["audio"],batch["audio_mask"],mean_only=self.variant=="mean")
  vv,pv=self.visual(batch["visual"],batch["visual_mask"],mean_only=self.variant=="mean")
  return dict(text=t,audio=aa,visual=vv,a_present=pa,v_present=pv)
 def components(self,batch):return self.components_from_repr(self.representations(batch))
 def components_from_repr(self,reps):
  t=reps["text"];base=self.text_head(t)
  if self.variant=="text":
   zero=torch.zeros_like(base);return dict(base=base,a=zero,v=zero,av=zero,audio_logits=zero,visual_logits=zero,a_present=torch.zeros(len(t),dtype=torch.bool,device=t.device),v_present=torch.zeros(len(t),dtype=torch.bool,device=t.device))
  aa=reps["audio"];vv=reps["visual"];pa=reps["a_present"];pv=reps["v_present"]
  ap=aa.mean(1);vp=vv.mean(1)
  ca,_=self.cross_audio(t[:,None],aa,aa,need_weights=False);cv,_=self.cross_visual(t[:,None],vv,vv,need_weights=False)
  ca=self.audio_norm(t+ca[:,0]);cv=self.visual_norm(t+cv[:,0])
  a=2*torch.tanh(self.audio_delta(torch.cat((t,ca,ap),-1))/2)*torch.sigmoid(self.audio_gate(torch.cat((t,ap),-1)))
  v=2*torch.tanh(self.visual_delta(torch.cat((t,cv,vp),-1))/2)*torch.sigmoid(self.visual_gate(torch.cat((t,vp),-1)))
  av=2*torch.tanh(self.joint_delta(torch.cat((t,ca,cv,ca*cv),-1))/2)*torch.sigmoid(self.joint_gate(torch.cat((t,ap,vp),-1)))
  return dict(base=base,a=a,v=v,av=av,audio_logits=self.audio_head(ap),visual_logits=self.visual_head(vp),a_present=pa,v_present=pv)
 def compose(self,c,mask):
  if self.variant=="text":return c["base"]
  m=mask.to(c["base"].device);a=m[:,0:1]*c["a_present"][:,None];v=m[:,1:2]*c["v_present"][:,None]
  return c["base"]+a*c["a"]+v*c["v"]+a*v*c["av"]
 def all_patterns(self,c):
  return torch.stack([self.compose(c,torch.tensor(bits,device=c["base"].device)[None].expand(len(c["base"]),-1)) for bits in PATTERN_BITS],dim=1)
 def forward(self,batch,mask=None):
  if mask is not None:
   mask=mask.to(batch["text"].device);batch=dict(batch)
   batch["audio_mask"]=batch["audio_mask"] & mask[:,0:1].bool()
   batch["visual_mask"]=batch["visual_mask"] & mask[:,1:2].bool()
  c=self.components(batch)
  if mask is None:mask=torch.ones((len(batch["text"]),2),device=c["base"].device)
  return self.compose(c,mask)
def objective(model,components,y,class_weights,uni_weight=.2,kd_weight=.1,temperature=2.,margin=.02):
 all_logits=model.all_patterns(components);zero=all_logits.sum()*0.
 if model.variant=="text":
  ce=F.cross_entropy(components["base"],y,weight=class_weights,label_smoothing=.05)
  return ce,dict(subset_ce=ce.detach(),unimodal_ce=zero.detach(),consistency=zero.detach(),teacher_selected=0.)
 weights=[.0225,.1275,.1275,.7225] if model.variant=="biased_subsets" else [.25]*4
 risk=sum(w*F.cross_entropy(all_logits[:,i],y,weight=class_weights,label_smoothing=.05) for i,w in enumerate(weights))
 unimodal=zero;terms=0
 if model.variant!="no_uni":
  for key,present in (("audio_logits",components["a_present"]),("visual_logits",components["v_present"])):
   if present.any():unimodal=unimodal+F.cross_entropy(components[key][present],y[present],weight=class_weights,label_smoothing=.05);terms+=1
  unimodal=unimodal/max(terms,1)
 teacher=all_logits[:,3].detach();confidence=teacher.softmax(-1).gather(1,y[:,None])[:,0]
 text_conf=all_logits[:,0].detach().softmax(-1).gather(1,y[:,None])[:,0]
 selected=(teacher.argmax(-1)==y)&(confidence>text_conf+margin);consistency=zero
 if model.variant!="no_consistency" and selected.any():
  target=(teacher/temperature).softmax(-1)
  student=F.log_softmax(all_logits[:,:3]/temperature,dim=-1)
  per=F.kl_div(student,target[:,None].expand(-1,3,-1),reduction="none").sum(-1).mean(1)*temperature**2
  consistency=per[selected].mean()
 loss=risk+uni_weight*unimodal+kd_weight*consistency
 return loss,dict(subset_ce=risk.detach(),unimodal_ce=unimodal.detach(),consistency=consistency.detach(),teacher_selected=float(selected.float().mean().detach().cpu()))
