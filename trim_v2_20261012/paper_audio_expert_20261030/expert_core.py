"""Mechanism-matched audio expert controls on the unchanged archived TRIM backbone."""
from pathlib import Path
import gc,hashlib,json,os,sys
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
FUSION=Path(os.environ.get('PAPER_FUSION_DIR',HERE.parent/'paper_audio_supervised_fusion_20261028')).resolve()
sys.path.insert(0,str(FUSION))
import fusion_core as previous
import run_fusion as previous_run
pc=previous.pc;info=previous.info
PARENT=previous.PARENT;CAUSAL=previous.CAUSAL;INFO=previous.INFO
MODES=previous.MODES;SEEDS=previous.SEEDS
predict=previous.predict
KINDS=('joint_repeat','stop_fusion_to_audio','frozen_residual','frozen_mixture','shuffled_mixture','zero_mixture')
MIXTURES=set(KINDS[-3:])

def verify_release():
 manifest=pc.read_json(HERE/'bundle_manifest.json')
 for rel,digest in manifest['files'].items():
  if pc.sha_file(HERE/rel)!=digest:raise RuntimeError('Expert source changed: '+rel)
 if pc.sha_file(FUSION/'bundle_manifest.json')!=manifest['fusion_manifest_sha256']:raise RuntimeError('Expected frozen supervised fusion release')
 previous.verify_release();return pc.sha_file(HERE/'bundle_manifest.json')

class ExpertFusion(previous.SupervisedFusion):
 def __init__(self,kind,input_dim=2048,text_dim=256):
  if kind not in KINDS:raise ValueError('Unknown expert condition')
  super().__init__(input_dim,text_dim);self.kind=kind
  self.frozen=kind not in ('joint_repeat','stop_fusion_to_audio')
  if self.frozen:self.audio.requires_grad_(False);self.audio.eval()
  if kind in MIXTURES:self.strength.requires_grad_(False)
 def train(self,mode=True):
  super().train(mode)
  if self.frozen:self.audio.eval()
  return self
 def forward(self,base_logits,ht,x,audio_present):
  base=base_logits.detach();a=self.audio(x.detach())
  context=torch.cat((ht.detach(),base.softmax(-1).detach(),a.softmax(-1).detach()),dim=-1)
  gate=torch.sigmoid(self.gate(context).squeeze(-1));present=audio_present.to(base.dtype)
  if self.kind in MIXTURES:
   # A fixed 0/1 switch preserves an exact baseline candidate; active gates train directly.
   weight=self.strength.clamp(0,1)*gate*present
   p=base.softmax(-1);q=a.softmax(-1).detach();mixed=p+weight[:,None]*(q-p)
   shift=mixed.clamp_min(1e-30).log()-p.clamp_min(1e-30).log()
   output=base+shift;delta=.5*(shift[:,1]-shift[:,0])
  else:
   expert=a.detach() if self.kind=='stop_fusion_to_audio' else a
   margin=.5*(expert[:,1]-expert[:,0])
   delta=torch.tanh(self.strength)*gate*2*torch.tanh(margin/2)*present
   output=base+torch.stack((-delta,delta),dim=-1)
  return output,a,delta,gate
 def activate(self):
  if self.kind in MIXTURES:
   with torch.no_grad():self.strength.fill_(1.)
 def project(self):pass
 def effective_strength(self):
  return float((self.strength.clamp(0,1) if self.kind in MIXTURES else torch.tanh(self.strength)).detach().cpu())

def assigned(data,kind,donors):
 if kind=='zero_mixture':return {s:np.zeros_like(data[s]['x']) for s in ('train','valid')}
 return {s:data[s]['x'][donors[s]] if kind=='shuffled_mixture' else data[s]['x'] for s in ('train','valid')}

def verify_prior(args):
 pipeline=pc.read_json(args.fusion_run/'pipeline_signature.json')
 if pipeline['release']!=pc.sha_file(FUSION/'bundle_manifest.json') or pipeline['test_evaluated']:raise RuntimeError('Previous fusion run differs')
 records={}
 for seed in SEEDS:
  folder=args.fusion_run/'full_supervised'/f'seed_{seed}';done=pc.read_json(folder/'complete.json')
  if done['status']!='complete' or done['signature']['pipeline']!=pipeline:raise RuntimeError('Incomplete previous full-supervised job')
  for rel,digest in done['artifacts'].items():
   if pc.sha_file(folder/rel)!=digest:raise RuntimeError('Previous fusion artifact changed: '+str(folder/rel))
  records[str(seed)]=dict(checkpoint_sha256=pc.sha_file(folder/'best.pt'),complete_sha256=pc.sha_file(folder/'complete.json'))
 return dict(pipeline_sha256=pc.sha_file(args.fusion_run/'pipeline_signature.json'),selected_joint_checkpoints=records,pipeline=pipeline)

def correction_capacity(y,base,audio,strength):
 wrong=base.argmax(1)!=y;required=np.abs((base[:,1]-base[:,0])/2)
 margin=(audio[:,1]-audio[:,0])/2;d=strength*2*np.tanh(margin/2)
 potential=base+np.stack((-d,d),axis=1)
 return dict(parent_errors=int(wrong.sum()),audio_correct_on_parent_errors=int((wrong&(audio.argmax(1)==y)).sum()),
  ideal_expert_reachable_under_scalar_bound=int((wrong&(required<2*abs(strength))).sum()),
  current_expert_reachable_with_gate_one=int((wrong&(potential.argmax(1)==y)).sum()),
  interpretation='diagnostic_reachability_counts_not_achievable_or_oracle_scores')

def gradient_rows(model,data,context,ce,device,seed,snapshot,batches=8,batch=128):
 # No optimizer. Real training rows only, fixed TAV/eval-mode snapshots; not historical stochastic gradients.
 model.eval();params=list(model.audio.parameters());rng=np.random.default_rng(20261030)
 chosen=rng.permutation(len(data['y']))[:min(len(data['y']),batches*batch)];rows=[]
 for start in range(0,len(chosen),batch):
  ix=chosen[start:start+batch]
  output,a,d,g=model(torch.as_tensor(context['logits_TAV'][ix],device=device),torch.as_tensor(context['ht'][ix],device=device),
   torch.as_tensor(data['x'][ix],device=device),torch.ones(len(ix),device=device,dtype=torch.bool))
  target=torch.as_tensor(data['y'][ix],device=device).long()
  fusion=ce(output,target);aux=ce(a,target);penalty=.001*d.square().mean()
  def vector(loss):
   grad=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
   return torch.cat([(torch.zeros_like(p) if v is None else v).reshape(-1) for p,v in zip(params,grad)])
  gf=vector(fusion);ga=vector(aux);gr=vector(penalty)
  nf=float(gf.norm().detach().cpu());na=float(ga.norm().detach().cpu());nr=float(gr.norm().detach().cpu())
  cosine=float((gf@ga/(gf.norm()*ga.norm()).clamp_min(1e-30)).detach().cpu()) if nf>1e-12 and na>1e-12 else None
  rows.append(dict(seed=seed,snapshot=snapshot,batch=start//batch,n=len(ix),fusion_grad_norm=nf,audio_grad_norm=na,
   weighted_audio_grad_norm=.2*na,residual_penalty_grad_norm=nr,cosine=cosine,
   fusion_to_weighted_aux_norm_ratio=nf/(.2*na+1e-30),scope='training_rows_eval_TAV_no_optimizer'))
 return rows

def diagnose(args,data,contexts,warm_states):
 counts=np.bincount(data['train']['y'],minlength=2);w=np.sqrt(counts.sum()/(2*counts));w=w/w.mean()
 ce=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=args.device),label_smoothing=.05)
 baseline={s:{m:info.metrics(data[s]['y'],contexts[s]['logits_'+m]) for m in MODES} for s in ('train','valid')}
 diagnostics=[];gradients=[]
 train_readouts=dict(ids=data['train']['ids'],labels=data['train']['y'],**{'parent_logits_'+m:contexts['train']['logits_'+m] for m in MODES})
 ix=np.random.default_rng(20261030).permutation(len(data['train']['y']))[:min(128,len(data['train']['y']))]
 pc.atomic_npz(args.out/'gradient_sample.npz',ids=data['train']['ids'][ix],indices=ix,labels=data['train']['y'][ix],
  x=data['train']['x'][ix],ht=contexts['train']['ht'][ix],logits_TAV=contexts['train']['logits_TAV'][ix])
 for seed,warm_state in warm_states.items():
  warm=ExpertFusion('joint_repeat').to(args.device);warm.audio.load_state_dict(warm_state,strict=True)
  tuned=ExpertFusion('joint_repeat').to(args.device)
  tuned.load_state_dict(torch.load(args.fusion_run/'full_supervised'/f'seed_{seed}/best.pt',map_location='cpu',weights_only=True),strict=True)
  gradients.extend(gradient_rows(warm,data['train'],contexts['train'],ce,args.device,seed,'initial_warm'))
  gradients.extend(gradient_rows(tuned,data['train'],contexts['train'],ce,args.device,seed,'previous_selected_joint'))
  for split in ('train','valid'):
   yy=data[split]['y'];base=contexts[split]['logits_TAV']
   a=info.predict(warm.audio,data[split]['x'],None,args.device);b=info.predict(tuned.audio,data[split]['x'],None,args.device)
   if split=='train':train_readouts['warm_audio_logits_'+str(seed)]=a;train_readouts['joint_audio_logits_'+str(seed)]=b
   diagnostics.append(dict(seed=seed,split=split,parent=baseline[split]['TAV'],warm_audio=info.metrics(yy,a),joint_audio=info.metrics(yy,b),
    warm_capacity=correction_capacity(yy,base,a,tuned.effective_strength()),tuned_capacity=correction_capacity(yy,base,b,tuned.effective_strength()),
    parent_prob_confidence_quantiles=np.quantile(np.maximum(info.prob(base),1-info.prob(base)),[0,.25,.5,.75,1]).tolist()))
  del warm,tuned;gc.collect()
  print('AUDIO_EXPERT_DIAGNOSED',seed,flush=True)
 pc.atomic_npz(args.out/'train_readouts.npz',**train_readouts)
 pc.write_csv(args.out/'gradient_diagnostics.csv',gradients)
 valid=[r for r in gradients if r['cosine'] is not None]
 result=dict(status='complete',baseline=baseline,audio=diagnostics,gradient_summary=dict(rows=len(gradients),nonzero_cosine_rows=len(valid),
  negative_cosine_rows=sum(r['cosine']<0 for r in valid),mean_cosine=float(np.mean([r['cosine'] for r in valid])) if valid else None),
  caveats=['Snapshot gradients on training data in eval/TAV mode, not a replay of historical dropout training gradients',
   'Warm and frozen TRIM experts were trained on these training rows; router training uses in-sample expert predictions, not out-of-fold predictions',
   'Validation used for read-only diagnosis only; no optimization in diagnose'],test_evaluated=False)
 pc.dump_json(args.out/'mechanism_diagnosis.json',result);return result