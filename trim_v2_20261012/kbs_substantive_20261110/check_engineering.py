"""Independent engineering tests. Synthetic results are never paper experiments."""
from pathlib import Path
import tempfile,json,hashlib,copy,random
import numpy as np
import torch
from common import *
from models import *
from interventions import *
from train import rng_state,restore_rng,save_torch
def main():
 runtime('cpu',17);checks={}
 # Full empirical projection is a no-op for additive scores and detects XOR interaction.
 n=8;aa=np.linspace(-1,1,n);bb=np.linspace(.3,-.4,n)
 projected,meta=exact_emap(n,lambda i,j:aa[i]+bb[j],pair_batch=7)
 assert np.max(np.abs(projected-aa-bb))<1e-12 and meta['grid_interaction_rms']<1e-7
 t=np.array([-1,-1,1,1]);v=np.array([-1,1,-1,1]);truth=t*v*2
 projected,meta=exact_emap(4,lambda i,j:2*t[i]*v[j],pair_batch=3)
 assert np.max(np.abs(projected))==0 and meta['grid_interaction_rms']==2
 checks['emap_additive_and_interaction_positive_control']=True
 anchor=np.array([0,2,4,6]);full,half,_=anchor_emap(n,anchor,lambda i,j:aa[i]+bb[j],pair_batch=5)
 assert np.max(np.abs(full-aa-bb))<1e-12 and np.max(np.abs(half-full))<1e-12
 checks['anchor_projection_preserves_additive_models']=True
 source=np.repeat(['a','b','c','d'],[3,3,2,2]);length=np.arange(10);edges=[2,4,6,8]
 for seed in range(10):
  d=cross_source(source,seed);assert np.all(source[d]!=source)and len(np.unique(d))==10
  d=within_source(source,seed);assert np.all(d!=np.arange(10))and np.all(source[d]==source)
  d,bin_=length_matched(source,length,edges,seed);assert np.all(d>=0)and np.all(source[d]!=source)and len(np.unique(d))==10
 assert np.all(cross_source(np.array(['one']*4),0)==-1)
 checks['donors_label_independent_bijections_and_coverage']=True
 # Projection and caches must preserve the actual legacy TRIM computation.
 identity={};t=torch.randint(0,64,(4,3,6));t[:,1]=1;t[:,2]=0;a=torch.randn(4,5);v=torch.randn(4,7)
 for variant in plan()['variants']:
  runtime('cpu',23);model,base=make_model('tiny',5,7,variant,d_model=32,tiny=True,return_legacy=True)
  model.eval();base.eval();identity[variant]=model.identity()
  with torch.inference_mode():
   z,_=model(t,a,v);ht=model.encode_text(t);ha,hv=model.encode_av(a,v)
   assert torch.allclose(z,model.fuse(ht,ha,hv),atol=1e-6)
   mask=torch.zeros((4,2),dtype=torch.bool)
   x,_=model(t,a,v,mask);xx,_=model(t,a*100+25,v*23,mask);assert torch.allclose(x,xx,atol=1e-6)
   if variant=='trim':
    original,_=base(t[:,0],t[:,1],t[:,2],a,v,training=False);assert torch.allclose(z,original,atol=2e-6)
    q,k=torch.randn(4,1,32),torch.randn(4,1,32)
    assert torch.allclose(model.gcm_ta(q,k),value_only_block(model.gcm_ta,q,k),atol=1e-6)
  if variant=='trim':
   model.zero_grad();model.gcm_ta(torch.randn(4,1,32),torch.randn(4,1,32)).square().sum().backward()
   grad=model.gcm_ta.cross_attn.in_proj_weight.grad;assert grad[:64].abs().max()<1e-7
  model.train();z,aux=model(t,a,v,torch.ones((4,2),dtype=torch.bool),auxiliary=True)
  loss=torch.nn.functional.cross_entropy(z,torch.tensor([0,1,0,1]))+.02*aux['nce']+.02*aux['ot'];loss.backward()
  assert all(x.grad is None or torch.isfinite(x.grad).all()for x in model.parameters())
  del model,base
 assert len({identity[k]['common_branch_initialization_sha256']for k in identity if k!='bert_mlp'})==1
 assert abs(identity['budget_mlp']['fusion_budget']['relative_parameter_difference'])<.01
 checks['five_variants_finite_backward_and_mask_invariance']=True
 checks['legacy_prediction_and_single_token_value_equivalence']=True
 checks['common_initialization_and_parameter_budget']=True
 # Resume must restore stochastic streams as well as model/optimizer tensors.
 runtime('cpu',19);net=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.3),torch.nn.Linear(8,2));opt=torch.optim.AdamW(net.parameters(),lr=.01);x=torch.randn(6,4)
 def step(net,opt):
  opt.zero_grad();loss=(net(x)-torch.rand(6,2)).square().mean();loss.backward();opt.step()
 step(net,opt);saved=dict(model=copy.deepcopy(net.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state());step(net,opt);want=copy.deepcopy(net.state_dict())
 net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);step(net,opt)
 assert all(torch.equal(want[k],net.state_dict()[k])for k in want)
 checks['resume_model_optimizer_and_rng_exact']=True
 y=np.array([0,1]*5);source=np.repeat(np.arange(5).astype(str),2);base=np.tile(np.array([-1,1]*5),(5,1));changed=base.copy();changed[:,0]=1
 first=paired_bootstrap(base,changed,y,source,np.ones(10,bool),128,5);second=paired_bootstrap(base,changed,y,source,np.ones(10,bool),128,5)
 assert first==second and first['delta_pp']<0
 checks['paired_hierarchical_bootstrap_reproducible']=True
 # Guard prevents held-out reading before all training checkpoints are sealed.
 with tempfile.TemporaryDirectory()as tmp:
  try:evaluation_data(tmp,'MOSI','test')
  except FileNotFoundError:pass
  else:raise AssertionError('Unsealed test data was readable')
 checks['unsealed_official_evaluation_rejected']=True
 report=dict(status='passed',checks=checks,identities=identity,formal_training_jobs=0,benchmark_inference_jobs=0,synthetic_engineering_only=True,
  torch=torch.__version__,numpy=np.__version__)
 dump(HERE/'checks/engineering_check.json',report);print('KBS_ENGINEERING_CHECK_PASSED',len(checks),'checks; no benchmark result claimed',flush=True)
if __name__=='__main__':main()