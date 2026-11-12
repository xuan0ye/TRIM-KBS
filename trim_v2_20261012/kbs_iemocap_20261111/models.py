"""Six-class systems sharing the legacy TRIM branches, BERT and paired training streams."""
from unittest.mock import patch
import hashlib,math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel,BertConfig
from common import HERE
import model_legacy as legacy
def count(module):return sum(p.numel()for p in module.parameters())
def head(dim,hidden=None,n_classes=6):
 hidden=hidden or dim//2
 return nn.Sequential(nn.LayerNorm(dim),nn.Dropout(.4),nn.Linear(dim,hidden),nn.GELU(),nn.Dropout(.2),nn.Linear(hidden,n_classes))
class ControlledModel(nn.Module):
 def __init__(self,base,variant):
  super().__init__();self.variant=variant;self.d=base.d;self.bert=base.bert;self.n_classes=6
  target=sum(count(getattr(base,n))for n in ['gcm_ta','gcm_at','gcm_tv','mm_ctx','classifier'])+base.mm_summary.numel()
  self.budget=dict(target_legacy_fusion_parameters=target)
  if variant=='bert_mlp':self.text_head=head(768);return
  for n in ['proj_t','proj_a','proj_v','uni_t','uni_a','uni_v','cproj_t','cproj_a','cproj_v','ot_ta']:
   setattr(self,n,getattr(base,n))
  self.missing_a=base.missing_a;self.missing_v=base.missing_v
  if variant=='trim':
   for n in ['gcm_ta','gcm_at','gcm_tv','mm_summary','mm_ctx','classifier']:setattr(self,n,getattr(base,n))
  elif variant=='concat':self.classifier=base.classifier
  elif variant=='additive':self.branch_heads=nn.ModuleList([head(self.d)for _ in range(3)])
  elif variant=='budget_mlp':
   dim=3*self.d
   width=min(range(16,4097),key=lambda h:abs(h*h+(dim+8)*h+2*dim+6-target))
   self.classifier=nn.Sequential(nn.LayerNorm(dim),nn.Dropout(.4),nn.Linear(dim,width),nn.GELU(),nn.Dropout(.2),nn.Linear(width,width),nn.GELU(),nn.Dropout(.2),nn.Linear(width,6))
   self.budget.update(width=width,actual_parameters=count(self.classifier),relative_parameter_difference=(count(self.classifier)-target)/target)
   assert abs(self.budget['relative_parameter_difference'])<.01
  else:raise ValueError(variant)
 def encode_text(self,t):
  x=self.bert(input_ids=t[:,0],attention_mask=t[:,1],token_type_ids=t[:,2],return_dict=True).last_hidden_state[:,0]
  return x if self.variant=='bert_mlp' else self.uni_t(self.proj_t(x)[:,None])[:,0]
 def encode_av(self,a,v,mask=None):
  if self.variant=='bert_mlp':return a.new_zeros((len(a),self.d)),v.new_zeros((len(v),self.d))
  a,v=self.proj_a(a),self.proj_v(v)
  if mask is not None:
   a=torch.where(mask[:,0,None],a,self.missing_a.expand(len(a),-1))
   v=torch.where(mask[:,1,None],v,self.missing_v.expand(len(v),-1))
  return self.uni_a(a[:,None])[:,0],self.uni_v(v[:,None])[:,0]
 def absent_states(self,n,device):
  if self.variant=='bert_mlp':return torch.zeros((n,self.d),device=device),torch.zeros((n,self.d),device=device)
  return self.uni_a(self.missing_a[:,None].expand(n,-1,-1))[:,0],self.uni_v(self.missing_v[:,None].expand(n,-1,-1))[:,0]
 def fuse(self,t,a,v):
  if self.variant=='bert_mlp':return self.text_head(t)
  if self.variant=='additive':return sum(h(x)for h,x in zip(self.branch_heads,[t,a,v]))/3
  if self.variant in ['concat','budget_mlp']:return self.classifier(torch.cat([t,a,v],-1))
  t,a,v=self.gcm_ta(t[:,None],a[:,None]),self.gcm_at(a[:,None],t[:,None]),self.gcm_tv(v[:,None],t[:,None])
  summ=self.mm_summary[None].expand(len(t),-1,-1)
  fused=self.mm_ctx(torch.cat([summ,t,a,v],1))[:,:len(self.mm_summary)].reshape(len(t),-1)
  return self.classifier(fused)
 def forward(self,t,a,v,mask=None,auxiliary=False):
  ht=self.encode_text(t);ha,hv=self.encode_av(a,v,mask);logits=self.fuse(ht,ha,hv)
  zero=logits.new_zeros(())
  if not auxiliary or self.variant=='bert_mlp':return logits,{'nce':zero,'ot':zero}
  z=[F.normalize(h(x),dim=-1)for h,x in zip([self.cproj_t,self.cproj_a,self.cproj_v],[ht,ha,hv])]
  nce=legacy.E2EMSAModel._sym_nce(self,z[0],z[1])+legacy.E2EMSAModel._sym_nce(self,z[0],z[2])
  ot=legacy.E2EMSAModel._sinkhorn(self,ht,ha)+legacy.E2EMSAModel._sinkhorn(self,ht,hv)
  return logits,{'nce':nce,'ot':ot}
 def identity(self):
  common=hashlib.sha256()
  for name,x in self.state_dict().items():
   if name.startswith(('proj_','uni_','missing_')):
    common.update(name.encode());common.update(x.detach().cpu().contiguous().numpy().tobytes())
  return dict(variant=self.variant,parameters=count(self),non_bert_parameters=sum(p.numel()for n,p in self.named_parameters()if not n.startswith('bert.')),common_branch_initialization_sha256=common.hexdigest(),fusion_budget=self.budget)
def make_model(bert,a_dim,v_dim,variant,pretrained=True,d_model=256,tiny=False,return_legacy=False):
 original=BertModel.from_pretrained
 def load(*args,**kw):
  if tiny:return BertModel(BertConfig(vocab_size=64,hidden_size=768,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64,max_position_embeddings=64))
  if not pretrained:
   c=BertConfig.from_pretrained(str(bert),local_files_only=True);c._attn_implementation='sdpa';return BertModel(c)
  m,info=original(str(bert),local_files_only=True,attn_implementation='sdpa',output_loading_info=True)
  if info['missing_keys']or info['mismatched_keys']or info['error_msgs']:raise RuntimeError('Incomplete BERT weight loading: '+str(info))
  return m
 with patch.object(legacy.BertModel,'from_pretrained',side_effect=load):base=legacy.E2EMSAModel(str(bert),a_dim,v_dim,d_model=d_model,n_classes=6)
 model=ControlledModel(base,variant)
 return (model,base)if return_legacy else model
def configure_phase(model,phase):
 for p in model.bert.parameters():p.requires_grad_(False)
 if phase==1:
  for layer in model.bert.encoder.layer[-2:]:
   for p in layer.parameters():p.requires_grad_(True)
  for p in model.bert.pooler.parameters():p.requires_grad_(True)
 if phase>=2:
  for p in model.bert.parameters():p.requires_grad_(True)
def optimizer(model,phase):
 from common import plan
 p=plan();lr=p['learning_rates'];bert_lr=0 if phase==0 else lr['bert_phase2'if phase==1 else'bert_phase3']
 groups={}
 for name,value in model.named_parameters():
  if not value.requires_grad:continue
  key='bert'if name.startswith('bert.')else('projector'if name.startswith('proj_')else'head')
  groups.setdefault(key,[]).append(value)
 rates={'bert':bert_lr,'projector':lr['phase1_projector'if phase==0 else'later_projector'],'head':lr['head']}
 return torch.optim.AdamW([{'params':v,'lr':rates[k],'weight_decay':p['weight_decay']}for k,v in groups.items()])
def value_only_block(block,q,source):
 d=q.shape[-1];m=block.cross_attn
 value=F.linear(source,m.in_proj_weight[2*d:],m.in_proj_bias[2*d:])
 value=m.out_proj(value);gate=block.gate(torch.cat([q,value],-1))
 h=block.norm1(q+block.drop(gate*value))
 return block.norm2(h+block.drop(block.ff(h)))