"""Protocol checks use a tiny stand-in for TRIM; no real MOSEI training is local."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack
import copy,hashlib,io,json,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_holdout as run
import holdout_core as core
from holdout_router import *
from holdout_split import make_plan

def expect_error(fn):
 try:fn()
 except (RuntimeError,ValueError):return
 raise AssertionError('Expected rejection')

class TinyTRIM(nn.Module):
 def __init__(self):super().__init__();self.d=256;self.classifier=nn.Linear(6,2)
 def forward(self,ids,mask,types,a,v,training=False):
  logits=self.classifier(a);zero=logits.sum()*0
  return logits,dict(gcl=zero,ot=zero,reg_ama=zero)

def main():
 pt.runtime('cpu',2);seed_all(51);rng=np.random.default_rng(51);checks={}
 scratch=HERE.parent.parent/'.tmp/holdout_check_20261030';scratch.mkdir(parents=True,exist_ok=True)
 work=Path(tempfile.mkdtemp(prefix='check_',dir=scratch));n=320
 ids=np.array(['video%d$_$%d'%(i//4,i) for i in range(n)]);y=np.arange(n)%2
 plan=make_plan(ids,y);flipped=make_plan(ids,1-y)
 assert all(plan['roles'][r]['indices']==flipped['roles'][r]['indices'] for r in ROLES)
 assert validate_plan(plan,ids,y)
 bad=copy.deepcopy(plan);bad['roles']['router_train']['indices'][0]=bad['roles']['expert_train']['indices'][0]
 expect_error(lambda:validate_plan(bad,ids,y));checks['complete_video_disjoint_label_independent_roles']=True
 with zipfile.ZipFile(HERE.parent/'audio_expert_review_20261030.zip') as z:
  with np.load(io.BytesIO(z.read('audio_expert_review/train_readouts.npz')),allow_pickle=False) as f:
   committed=pc.read_json(HERE/'split_plan.json');assert validate_plan(committed,f['ids'],f['labels'])
   for role in ('router_train','router_dev'):
    ix=committed['roles'][role]['indices'];cross_video(f['ids'][ix],20261030)
 checks['committed_real_split_identity_and_permutation_feasibility']=True
 text=np.zeros((n,3,6),np.int64);text[:,0]=rng.integers(1,20,(n,6));text[:,1]=1
 paper=dict(ids=ids,y=y,text=text,audio=rng.normal(0,.4,(n,6)).astype(np.float32),vision=rng.normal(0,.2,(n,5)).astype(np.float32))
 paper['audio'][:,0]=(2*y-1)*.6+rng.normal(0,.5,n)
 embeddings=rng.normal(0,.2,(n,2048)).astype(np.float32);embeddings[:,0]=2*(2*y-1)
 data=partition_training(paper,embeddings,plan,work/'normalization.npz')
 changed=embeddings.copy();outside=np.concatenate([plan['roles'][r]['indices'] for r in ROLES if r!='expert_train']);changed[outside]+=100000
 partition_training(paper,changed,plan,work/'changed_normalization.npz')
 a=pc.load_npz(work/'normalization.npz');b=pc.load_npz(work/'changed_normalization.npz')
 assert all(np.array_equal(a[k],b[k]) for k in a)
 checks['normalization_uses_expert_train_only']=True
 parts=expert_parts(data)
 assert np.array_equal(parts['train']['ids'],data['expert_train']['paper']['ids']) and np.array_equal(parts['valid']['ids'],data['expert_dev']['paper']['ids'])
 checks['paper_trainer_receives_only_expert_train_and_dev']=True
 for constant in (False,True):
  model=ProbabilityRouter(constant);assert sum(p.numel() for p in model.parameters())==(1 if constant else 145)
  base=torch.randn(8,2,requires_grad=True);audio=torch.randn(8,2,requires_grad=True)
  logits,w=model(base,audio,torch.ones(8,dtype=torch.bool));assert torch.equal(logits,base)
  model.activate();logits,w=model(base,audio,torch.zeros(8,dtype=torch.bool));assert torch.equal(logits,base)
  logits,w=model(base,audio,torch.ones(8,dtype=torch.bool));nn.functional.cross_entropy(logits,torch.arange(8)%2).backward()
  assert base.grad is None and audio.grad is None and sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)>0
 checks['router_capacity_identity_missing_audio_and_expert_gradient_guard']=True
 # Entire stage-order smoke with the unchanged real paper trainer but a lightweight test stand-in.
 args=SimpleNamespace(out=work/'pipeline',root=work,parent_run=work/'parent',device='cpu',seed=42,batch=8,acc_steps=4,router_epochs=8,permutations=3)
 args.out.mkdir();events=[];phase_calls=[];created=[]
 external_n=32;ey=np.arange(external_n)%2
 external=dict(paper=dict(ids=np.array(['external%d$_$%d'%(i//4,i) for i in range(external_n)]),y=ey,
  text=np.tile(text[0:1],(external_n,1,1)),audio=rng.normal(0,.4,(external_n,6)).astype(np.float32),vision=rng.normal(0,.2,(external_n,5)).astype(np.float32)),
  x=rng.normal(0,.3,(external_n,2048)).astype(np.float32))
 reference=args.parent_run/'paper_original/seed_42';reference.mkdir(parents=True)
 pc.atomic_npz(reference/'validation_predictions.npz',ids=external['paper']['ids'],labels=ey,pred_TAV=np.zeros(external_n,np.int64),prob_TAV=np.full(external_n,.4,np.float32))
 def prep(a):
  events.append('prepare_training');normal=pc.load_npz(work/'normalization.npz');keep_stats(a.out/'normalization.npz',normal['mean'],normal['std'])
  pc.atomic_npz(a.out/'role_index.npz',ids=ids,labels=y,**{r:np.asarray(plan['roles'][r]['indices']) for r in ROLES})
  pc.dump_json(a.out/'split_plan.json',plan)
  return data,dict(synthetic=True,normalization_sha256=pc.sha_file(a.out/'normalization.npz'))
 def bert(a,b):
  events.append('verify_original_pretrained');pc.dump_json(b/'bert_verified.json',dict(synthetic_original_pretrained=True));return work/'original_pretrained'
 def make(bert,a,v,pretrained=True):
  created.append(pretrained);assert pretrained is True;events.append('fresh_paper_expert');return TinyTRIM()
 def evaluate(model,part,batch,device,all_masks=False):
  assert set(part['ids'])==set(data['expert_dev']['paper']['ids']);events.append('select_expert_dev')
  model.eval()
  with torch.inference_mode():logits=model.classifier(torch.tensor(part['audio']))
  q=logits.softmax(-1)[:,1].numpy();p=logits.argmax(1).numpy();m=pc.scores(part['y'],p,q)
  modes=MODES if all_masks else ('TAV',)
  return {**{'pred_'+k:p for k in modes},**{'prob_'+k:q for k in modes}},{k:m for k in modes},dict(forward_parity_max_delta=0.,absent_audio_max_delta=0.)
 def load(a,bert,dd):
  model=TinyTRIM();model.load_state_dict(torch.load(a.out/'trim_expert/paper_original/seed_42/best.pt',map_location='cpu',weights_only=True));return model.eval().requires_grad_(False)
 def final_external(a):
  events.append('external_loaded');assert (a.out/'selection_seal.json').is_file()
  seal=pc.read_json(a.out/'selection_seal.json')
  assert all(pc.sha_file(a.out/'routers'/k/'best.pt')==seal['routers'][k] for k in CONDITIONS)
  return external
 def freeze(model):phase_calls.append('freeze_all')
 def last(model,n,verbose=False):assert n==2;phase_calls.append('unfreeze_last_two')
 def all_layers(model,verbose=False):phase_calls.append('unfreeze_all')
 def optimizer(model,lr_b,lr_p,lr_aux):return torch.optim.AdamW(model.parameters(),lr=lr_p)
 with ExitStack() as stack:
  for owner,name,value in ((run,'verify_release',lambda:'synthetic_release'),(run,'prepare_training',prep),(run,'preflight_bert',bert),
   (run.pt,'make_model',make),(run.pt,'evaluate',evaluate),(run,'load_trim',load),(run,'prepare_external',final_external),
   (run.pt.legacy,'freeze_bert_all',freeze),(run.pt.legacy,'unfreeze_bert_last_n',last),(run.pt.legacy,'unfreeze_bert_all',all_layers),(run.pt.legacy,'build_optimizer',optimizer),
   (run.pt,'encode',lambda model,tb,ta,tv:(ta,ta,tv)),(run.pt,'fuse',lambda model,ht,ha,hv,mask:model.classifier(ht))):
   stack.enter_context(patch.object(owner,name,side_effect=value))
  run.run_pipeline(args)
  assert created==[True] and phase_calls==['freeze_all','unfreeze_last_two','unfreeze_all']
  for kind in CONDITIONS:
   prediction=pc.load_npz(args.out/'routers'/kind/'router_dev_predictions.npz')
   expected=cross_video(data['router_dev']['paper']['ids'],20261031) if kind=='gated_shuffled' else np.arange(len(data['router_dev']['x']))
   assert np.array_equal(prediction['donors'],expected)
  assert events.index('external_loaded')>events.index('fresh_paper_expert')
  summary=pc.read_json(args.out/'summary.json');assert summary['external']['selection_sealed_before_external']
  selected={k:pc.sha_file(args.out/'routers'/k/'best.pt') for k in CONDITIONS}
  run.run_pipeline(args);assert created==[True] and all(pc.sha_file(args.out/'routers'/k/'best.pt')==v for k,v in selected.items())
 checks['fresh_20_epoch_paper_stages_and_internal_expert_selection']=True
 checks['end_to_end_internal_router_selection_then_sealed_external_evaluation']=True
 checks['completed_pipeline_resume_preserves_all_selected_weights']=True
 archive=args.out/'audio_holdout_review_20261030.zip'
 with zipfile.ZipFile(archive) as z:
  manifest=json.loads(z.read('audio_holdout_review/review_manifest.json'))
  for f,digest in manifest['files'].items():assert hashlib.sha256(z.read('audio_holdout_review/'+f)).hexdigest()==digest
  assert not any('trim_expert' in name and name.endswith('.pt') for name in z.namelist())
 checks['review_archive_hashes_and_large_backbone_exclusion']=True
 # Seal rejects a changed selected model before external inference.
 path=args.out/'routers/gated_correct/best.pt'
 with path.open('ab') as f:f.write(b'changed')
 expect_error(lambda:external_evaluation(args,pc.load_npz(args.out/'readouts/external_valid.npz'),summary['router_jobs'],summary['selections']))
 checks['changed_selection_rejected_after_seal']=True
 # Known useful audio signal on synthetic expert-unseen posterior rows.
 known={}
 for role,count in (('train',320),('dev',80)):
  yy=np.tile([0,1],count//2);pred=yy.copy();bad=rng.choice(count,count//3,replace=False);pred[bad]=1-pred[bad]
  p=np.where(pred==1,.65,.35).astype(np.float32);q=np.where(yy==1,.99,.01).astype(np.float32)
  ll=np.stack((np.log1p(-p),np.log(p)),1);aa=np.stack((np.log1p(-q),np.log(q)),1)
  known[role]=dict(ids=np.array(['%s%d$_$%d'%(role,i//4,i) for i in range(count)]),y=yy,audio_logits=aa,zero_audio_logits=np.zeros_like(aa),**{'logits_'+m:ll.copy() for m in MODES})
 ka=SimpleNamespace(out=work/'known_signal',seed=42,router_epochs=50,device=torch.device('cpu'));ka.out.mkdir()
 result=train_router(ka,'gated_correct',known['train'],known['dev'],dict(synthetic_known_signal=True))
 assert result['best_router_dev_macro_f1']>.95 and result['best_epoch']>0
 checks['router_learns_known_complementary_signal']=True
 tested=('holdout_split.py','holdout_core.py','holdout_router.py','run_holdout.py','checks/check_holdout.py')
 pc.dump_json(HERE/'checks/engineering_check.json',dict(status='pass',scope='synthetic_protocol_smoke_tiny_TRIM_no_real_MOSEI_optimization',
  checks=checks,tested_source_sha256={f:pc.sha_file(HERE/f) for f in tested},real_split_role_counts={r:pc.read_json(HERE/'split_plan.json')['roles'][r]['n'] for r in ROLES},
  torch=torch.__version__,numpy=np.__version__,work=str(work),test_evaluated=False))
 print('HOLDOUT_ENGINEERING_PASS',len(checks),flush=True)
if __name__=='__main__':main()