"""Exact legacy inference plus train-split-only diagnostic probes."""
from pathlib import Path
from unittest.mock import patch
import json,math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertConfig,BertModel
import model_legacy as legacy
def runtime(device="auto",threads=4):
 torch.backends.cudnn.enabled=False
 torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False
 torch.backends.cudnn.allow_tf32=False
 torch.set_num_threads(threads)
 torch.set_default_dtype(torch.float32)
 if device=="auto":device="cuda:0" if torch.cuda.is_available() else "cpu"
 return torch.device(device)
def make_model(bert_path,a_dim,v_dim):
 config=BertConfig.from_pretrained(str(bert_path),local_files_only=True)
 config._attn_implementation="sdpa"
 with patch.object(legacy.BertModel,"from_pretrained",side_effect=lambda *a,**kw:BertModel(config)):
  return legacy.E2EMSAModel(str(bert_path),a_dim,v_dim,d_model=256,n_classes=2)
def configure_variant(model,variant):
 model.use_audio=variant!="text_only";model.use_visual=variant!="text_only"
 model.modality_dropout_p=.15 if variant=="full" else 0.
 model.use_contrastive=True;model.use_ot=True;model.use_ama=True
def encode(model,tb,ta,tv):
 out=model.bert(input_ids=tb[:,0],attention_mask=tb[:,1],token_type_ids=tb[:,2],return_dict=True)
 return model.proj_t(out.last_hidden_state[:,0]),model.proj_a(ta),model.proj_v(tv)
def fuse(model,ht,ha,hv,mask,return_tokens=False):
 b=len(ht);mask=mask.to(ht.device).clone()
 if not model.use_audio:mask[:,1]=0
 if not model.use_visual:mask[:,2]=0
 data=[]
 for i,(h,placeholder) in enumerate(zip((ht,ha,hv),(model.missing_t,model.missing_a,model.missing_v))):
  fill=placeholder.expand(b,-1) if model.use_missing_repr else torch.zeros_like(h)
  data.append(torch.where(mask[:,i,None]>.5,h,fill))
 ht,ha,hv=data
 if model.use_uni_ctx:ht,ha,hv=model.uni_t(ht[:,None]),model.uni_a(ha[:,None]),model.uni_v(hv[:,None])
 else:ht,ha,hv=ht[:,None],ha[:,None],hv[:,None]
 if model.use_gcm:t,a,v=model.gcm_ta(ht,ha),model.gcm_at(ha,ht),model.gcm_tv(hv,ht)
 else:t,a,v=ht,ha,hv
 if model.use_mm_ctx:
  if model.use_summary_token:
   summ=model.mm_summary[None].expand(b,-1,-1)
   seq=model.mm_ctx(torch.cat([summ,t,a,v],dim=1));merged=seq[:,:model.n_summary].reshape(b,-1)
  else:merged=model.mm_ctx(torch.cat([t,a,v],dim=1)).reshape(b,-1)
 else:merged=torch.cat([t[:,0],a[:,0],v[:,0]],dim=-1)
 logits=model.classifier(merged)
 return (logits,dict(text=ht,audio=ha,vision=hv)) if return_tokens else logits
def balanced_probe_indices(labels,seed,n=8):
 rng=np.random.default_rng(seed);half=n//2
 out=np.r_[rng.choice(np.flatnonzero(labels==0),half,replace=False),rng.choice(np.flatnonzero(labels==1),n-half,replace=False)]
 return rng.permutation(out)
def norm(grad):
 return None if grad is None else float(torch.linalg.vector_norm(grad.detach().float()).cpu())
def diagnostic_probe(model,train,seed,device):
 # No optimization or selection; the deterministic probe uses TRAIN samples only.
 model.eval()
 for p in model.bert.parameters():p.requires_grad_(False)
 ix=balanced_probe_indices(train["y"],seed)
 tb=torch.as_tensor(train["text"][ix],device=device)
 ta=torch.tensor(train["audio"][ix],device=device,requires_grad=True)
 tv=torch.tensor(train["vision"][ix],device=device,requires_grad=True)
 y=torch.as_tensor(train["y"][ix],dtype=torch.long,device=device)
 counts=np.bincount(train["y"],minlength=2)
 w=np.sqrt(counts.sum()/(2*np.maximum(counts,1)));w=w/w.mean()
 w=torch.as_tensor(w,dtype=torch.float32,device=device)
 named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
 pars=[p for _,p in named]+[ta,tv]
 logits,aux=model(tb[:,0],tb[:,1],tb[:,2],ta,tv,training=False)
 ce=F.cross_entropy(logits,y,weight=w,label_smoothing=.05)
 auxloss=.02*aux["gcl"]+.02*aux["ot"]+.001*aux["reg_ama"]
 cg=torch.autograd.grad(ce,pars,retain_graph=True,allow_unused=True)
 ag=torch.autograd.grad(auxloss,pars,allow_unused=True)
 cdict={n:g for (n,_),g in zip(named,cg)};adict={n:g for (n,_),g in zip(named,ag)}
 def group_norm(values,prefix):
  gg=[g for n,g in values.items() if n.startswith(prefix) and g is not None]
  return math.sqrt(sum(float(g.detach().double().square().sum().cpu()) for g in gg)) if gg else None
 prefixes=("proj_t.","proj_a.","proj_v.","uni_t.","uni_a.","uni_v.","gcm_ta.","gcm_at.","gcm_tv.","mm_ctx.","classifier.","ot_ta.","ot_tv.","cproj_t.","cproj_a.","cproj_v.","missing_a","missing_v","task_priority","task_log_vars")
 qkv={}
 for name in ("gcm_ta","gcm_at","gcm_tv"):
  g=cdict.get(name+".cross_attn.in_proj_weight");d=model.d
  qkv[name]=dict(query=norm(g[:d]) if g is not None else None,key=norm(g[d:2*d]) if g is not None else None,value=norm(g[2*d:]) if g is not None else None)
 for g in list(cg)+list(ag):
  assert g is None or torch.isfinite(g).all(),"Nonfinite diagnostic gradient"
 # Observe legacy auxiliary losses when both A/V inputs are replaced by placeholders.
 previous=model.modality_dropout_p;model.modality_dropout_p=1.
 _,missing_aux=model(tb[:,0],tb[:,1],tb[:,2],ta.detach(),tv.detach(),training=True)
 mloss=.02*missing_aux["gcl"]+.02*missing_aux["ot"]
 mg=torch.autograd.grad(mloss,[model.missing_a,model.missing_v,model.ot_tv.weight,model.task_priority],allow_unused=True)
 model.modality_dropout_p=previous
 opt=legacy.build_optimizer(model,0.,5e-5,3e-4)
 inopt={id(p) for group in opt.param_groups for p in group["params"]}
 coverage={prefix:all(id(p) in inopt for n,p in named if n.startswith(prefix)) for prefix in ("proj_a.","proj_v.","uni_a.","uni_v.","gcm_ta.","gcm_at.","gcm_tv.")}
 return dict(probe_split="train",probe_seed=seed,n_probe=len(ix),sample_ids=train["ids"][ix].tolist(),
  model_mode="eval; gradients enabled; BERT frozen; no optimizer step",
  ce=float(ce.detach().cpu()),contrastive=float(aux["gcl"].detach().cpu()),ot=float(aux["ot"].detach().cpu()),
  ce_group_grad_l2={p:group_norm(cdict,p) for p in prefixes},
  aux_group_grad_l2={p:group_norm(adict,p) for p in prefixes},
  ce_input_grad_l2=dict(audio=norm(cg[-2]),vision=norm(cg[-1])),gcm_qkv_ce_grad_l2=qkv,
  optimizer_includes_av_parameters=coverage,
  all_av_absent_aux=dict(contrastive=float(missing_aux["gcl"].detach().cpu()),ot=float(missing_aux["ot"].detach().cpu()),missing_audio_grad=norm(mg[0]),missing_visual_grad=norm(mg[1]),unused_ot_tv_grad=norm(mg[2]),unused_task_priority_grad=norm(mg[3])),
  warning="Gradient connectivity diagnoses this checkpoint graph, not the history of optimizer updates. Gate values alone do not establish modality importance.")
