"""Synthetic CPU mechanism checks; no real MOSEI optimization."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,io,json,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_expert as run
from expert_core import *

def expect_error(fn):
 try:fn()
 except (ValueError,RuntimeError):return
 raise AssertionError('Expected integrity rejection')

def main():
 info.runtime('cpu',2);torch.manual_seed(23);rng=np.random.default_rng(23);checks={}
 scratch=HERE.parent.parent/'.tmp/expert_engineering_20261030';scratch.mkdir(parents=True,exist_ok=True)
 work=Path(tempfile.mkdtemp(prefix='check_',dir=scratch))
 base=torch.randn(10,2);ht=torch.randn(10,8);x=torch.randn(10,16);target=torch.arange(10)%2
 for kind in KINDS:
  m=ExpertFusion(kind,16,8).eval();out,a,d,g=m(base,ht,x,torch.ones(10,dtype=torch.bool))
  assert torch.equal(out,base) and (d==0).all()
  m.strength.data.fill_(.7);out,a,d,g=m(base,ht,x,torch.zeros(10,dtype=torch.bool));assert torch.equal(out,base)
 checks['all_six_exact_identity_and_missing_audio']=True
 for kind in KINDS:
  real=ExpertFusion(kind);trainable=sum(p.numel() for p in real.parameters() if p.requires_grad)
  assert trainable==(74084 if kind in ('joint_repeat','stop_fusion_to_audio') else 8385 if kind in MIXTURES else 8386)
 checks['trainable_parameter_counts']=True
 # Exact equivalence to the previous joint form, including parameter gradients.
 old=previous.SupervisedFusion(16,8).eval();old.strength.data.fill_(.4)
 new=ExpertFusion('joint_repeat',16,8).eval();new.load_state_dict(old.state_dict())
 old_out=old(base,ht,x,torch.ones(10,dtype=torch.bool));new_out=new(base,ht,x,torch.ones(10,dtype=torch.bool))
 assert all(torch.equal(a,b) for a,b in zip(old_out,new_out))
 nn.functional.cross_entropy(old_out[0],target).backward();nn.functional.cross_entropy(new_out[0],target).backward()
 for a,b in zip(old.parameters(),new.parameters()):
  assert (a.grad is None and b.grad is None) or torch.equal(a.grad,b.grad)
 checks['joint_repeat_matches_previous_outputs_and_gradients']=True
 stopped=ExpertFusion('stop_fusion_to_audio',16,8).eval();stopped.strength.data.fill_(.4)
 output,a,d,g=stopped(base,ht,x,torch.ones(10,dtype=torch.bool));nn.functional.cross_entropy(output,target).backward()
 assert all(p.grad is None or p.grad.abs().sum()==0 for p in stopped.audio.parameters())
 stopped.zero_grad(set_to_none=True);output,a,d,g=stopped(base,ht,x,torch.ones(10,dtype=torch.bool));nn.functional.cross_entropy(a,target).backward()
 assert stopped.audio.net[0].weight.grad.abs().sum()>0
 checks['stopped_fusion_gradient_and_retained_audio_supervision']=True
 frozen=ExpertFusion('frozen_mixture',16,8);frozen.train()
 assert frozen.training and not frozen.audio.training and all(not p.requires_grad for p in frozen.audio.parameters())
 aa=frozen.audio(x);bb=frozen.audio(x);assert torch.equal(aa,bb)
 frozen.activate();out,a,d,g=frozen(base,ht,x,torch.ones(10,dtype=torch.bool));nn.functional.cross_entropy(out,target).backward()
 assert frozen.gate[-1].weight.grad.abs().sum()>0 and all(p.grad is None for p in frozen.audio.parameters())
 assert not frozen.strength.requires_grad and frozen.strength.grad is None and float(frozen.strength)==1.
 checks['frozen_eval_mode_determinism_active_gate_gradient_and_fixed_switch']=True
 # A mixture can replace an overconfident wrong parent; no test labels enter this forward pass.
 frozen=ExpertFusion('frozen_mixture',16,8).eval()
 with torch.no_grad():
  frozen.strength.fill_(1.);frozen.gate[-1].weight.zero_();frozen.gate[-1].bias.fill_(20.)
  frozen.audio.net[-1].weight.zero_();frozen.audio.net[-1].bias.copy_(torch.tensor([-4.,4.]))
 hard=torch.tensor([[3.,-3.]]).expand(10,-1)
 logits,a,d,g=frozen(hard,ht,x,torch.ones(10,dtype=torch.bool))
 assert (logits.argmax(1)==1).all() and torch.max(torch.abs(logits.softmax(1)-a.softmax(1)))<1e-6
 checks['probability_mixture_can_override_confident_wrong_parent']=True
 # Real warmstart weights must fit all six complete 2048D expert interfaces.
 with zipfile.ZipFile(HERE.parent/'audio_information_review_20261028.zip') as z:
  for seed in SEEDS:
   weights=torch.load(io.BytesIO(z.read(f'audio_information_review/panns_embedding/seed_{seed}/best.pt')),map_location='cpu',weights_only=True)
   for kind in KINDS:
    m=ExpertFusion(kind).eval();m.audio.load_state_dict(weights,strict=True)
 checks['three_real_warmstarts_all_six_architectures_strict_load']=True
 data={};contexts={}
 for split,n in (('train',160),('valid',64)):
  y=np.tile([0,1],n//2).astype(np.int64);rng.shuffle(y);xx=rng.normal(0,.2,(n,16)).astype(np.float32)
  xx[:,0]=2*(2*y-1)+rng.normal(0,.15,n);p=y.copy();bad=rng.choice(n,n//3,replace=False);p[bad]=1-p[bad]
  margin=(2*p-1).astype(np.float32)*.3;logits=np.stack((-margin,margin),1)
  data[split]=dict(ids=np.array(['%s%d$_$%d'%(split,i//4,i) for i in range(n)]),y=y,x=xx)
  contexts[split]=dict(y=y,ht=rng.normal(0,.2,(n,8)).astype(np.float32),**{'logits_'+m:logits.copy() for m in MODES})
 donors={s:info.cross_video(data[s]['ids'],23+i) for i,s in enumerate(('train','valid'))}
 warm=info.Probe('panns_embedding',16);opt=torch.optim.AdamW(warm.parameters(),lr=.01)
 for step in range(65):
  warm.train();opt.zero_grad();loss=nn.functional.cross_entropy(warm(torch.tensor(data['train']['x'])),torch.tensor(data['train']['y']))
  loss.backward();opt.step()
 weights=run.state(warm);identity=dict(checkpoint_sha256=run.fingerprint(warm),synthetic=True)
 m=ExpertFusion('joint_repeat',16,8);m.audio.load_state_dict(weights)
 ce=nn.CrossEntropyLoss(label_smoothing=.05)
 grad=gradient_rows(m,data['train'],contexts['train'],ce,torch.device('cpu'),13,'synthetic_initial',batches=2,batch=64)
 assert len(grad)==2 and all(r['fusion_grad_norm']==0 and r['weighted_audio_grad_norm']>0 for r in grad)
 checks['training_gradient_snapshot_zero_at_identity']=True
 args=SimpleNamespace(out=work/'runs',epochs=12,batch=32,device=torch.device('cpu'),audio_lr=.001,fusion_lr=.03,permutations=3,seeds=[13],mode='run')
 args.out.mkdir();pc.dump_json(args.out/'pipeline_signature.json',dict(synthetic=True,test_evaluated=False))
 pc.atomic_npz(args.out/'validation_features.npz',ids=data['valid']['ids'],labels=data['valid']['y'],x=data['valid']['x'],ht=contexts['valid']['ht'],
  **{'logits_'+m:contexts['valid']['logits_'+m] for m in MODES})
 jobs=[run.fit(args,data,contexts,k,13,donors,weights,identity) for k in KINDS]
 assert len({j['identity']['initialization_sha256'] for j in jobs})==1
 assert all(j['missing_audio_exact_identity'] for j in jobs)
 assert all(j['frozen_audio_hash_unchanged'] for j in jobs if j['kind'] not in ('joint_repeat','stop_fusion_to_audio'))
 mix=next(j for j in jobs if j['kind']=='frozen_mixture');zero=next(j for j in jobs if j['kind']=='zero_mixture')
 assert mix['modes']['TAV']['macro_f1']>.95 and mix['modes']['TAV']['macro_f1']>zero['modes']['TAV']['macro_f1']
 assert mix['assigned_minus_permuted_pp']>20
 checks['six_training_controls_shared_initialization_frozen_expert_and_known_signal']=True
 assert run.fit(args,data,contexts,KINDS[0],13,donors,weights,identity)==jobs[0]
 run.summarize(args,jobs,dict(status='synthetic_only'))
 with zipfile.ZipFile(args.out/'audio_expert_review_20261030.zip') as z:
  manifest=json.loads(z.read('audio_expert_review/review_manifest.json'))
  for f,digest in manifest['files'].items():assert hashlib.sha256(z.read('audio_expert_review/'+f)).hexdigest()==digest
 checks['resume_and_review_archive_all_hashes']=True
 path=args.out/KINDS[0]/'seed_13/best.pt'
 with path.open('ab') as f:f.write(b'changed')
 expect_error(lambda:run.fit(args,data,contexts,KINDS[0],13,donors,weights,identity))
 checks['completed_checkpoint_mutation_rejected']=True
 pc.dump_json(HERE/'checks/engineering_check.json',dict(status='pass',scope='synthetic_CPU_no_real_MOSEI_optimization',checks=checks,
  synthetic_results=[dict(kind=j['kind'],f1=j['modes']['TAV']['macro_f1'],best_epoch=j['best_epoch']) for j in jobs],
  tested_source_sha256={f:pc.sha_file(HERE/f) for f in ('expert_core.py','run_expert.py','checks/check_expert.py')},
  torch=torch.__version__,numpy=np.__version__,work=str(work),test_evaluated=False))
 print('EXPERT_ENGINEERING_PASS',len(checks),flush=True)
if __name__=='__main__':main()