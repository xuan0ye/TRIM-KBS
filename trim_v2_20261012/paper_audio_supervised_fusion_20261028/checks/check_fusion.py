"""CPU engineering checks on synthetic data; not MOSEI improvement evidence."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,io,json,sys,tempfile,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_fusion as run
from fusion_core import *

def expect_error(fn):
 try:fn()
 except (RuntimeError,ValueError):return
 raise AssertionError('Expected integrity rejection')

def main():
 info.runtime('cpu',2);torch.manual_seed(17);rng=np.random.default_rng(17)
 root=HERE.parent.parent/'.tmp/fusion_engineering_20261028';root.mkdir(parents=True,exist_ok=True)
 work=Path(tempfile.mkdtemp(prefix='check_',dir=root));checks={}
 model=SupervisedFusion(16,8)
 base=torch.randn(12,2,requires_grad=True);ht=torch.randn(12,8,requires_grad=True);xx=torch.randn(12,16,requires_grad=True)
 out,a,d,g=model(base,ht,xx,torch.ones(12,dtype=torch.bool))
 assert torch.equal(out,base) and (d==0).all()
 loss=out.square().mean()+.2*a.square().mean();loss.backward()
 assert base.grad is None and ht.grad is None and xx.grad is None
 assert model.strength.grad.abs()>0 and model.audio.net[0].weight.grad.abs().sum()>0
 checks['exact_initial_identity_and_frozen_input_gradient_guard']=True
 model.strength.data.fill_(1.)
 out,a,d,g=model(base,ht,xx,torch.zeros(12,dtype=torch.bool));assert torch.equal(out,base) and torch.equal(d,torch.zeros_like(d))
 checks['missing_audio_exact_identity_nonzero_head']=True
 assert sum(p.numel() for p in SupervisedFusion().parameters())==74084
 checks['real_parameter_count']=74084
 with zipfile.ZipFile(HERE.parent/'audio_information_review_20261028.zip') as z:
  for seed in SEEDS:
   weights=torch.load(io.BytesIO(z.read('audio_information_review/panns_embedding/seed_'+str(seed)+'/best.pt')),map_location='cpu',weights_only=True)
   fused=SupervisedFusion().eval();fused.audio.load_state_dict(weights,strict=True)
   alone=info.Probe('panns_embedding',2048).eval();alone.load_state_dict(weights,strict=True)
   rx=torch.randn(5,2048);rt=torch.randn(5,256);rb=torch.randn(5,2)
   with torch.inference_mode():
    output,a,d,g=fused(rb,rt,rx,torch.ones(5,dtype=torch.bool))
    assert torch.equal(output,rb) and torch.equal(a,alone(rx))
 checks['three_returned_real_audio_checkpoints_strict_2048D_compatibility']=True
 expected=dict(version=1);path=work/'cache.npz'
 pc.atomic_npz(path,y=np.array([0,1]));pc.dump_json(path.with_suffix('.json'),dict(signature=expected,sha256=pc.sha_file(path)))
 assert read_cache(path,expected)[0]['y'].tolist()==[0,1]
 expect_error(lambda:read_cache(path,dict(version=2)))
 with path.open('ab') as f:f.write(b'changed')
 expect_error(lambda:read_cache(path,expected));checks['cache_signature_and_hash_guard']=True
 # Real 2048D feature loader must reject tampering; this is synthetic only.
 mock=work/'information';mock.mkdir();ids=np.array(['a','b']);norm={}
 pc.atomic_npz(mock/'dataset_valid.npz',ids=ids,y=np.array([0,1]),panns_embedding=np.zeros((2,2048),np.float32))
 expected_inputs={'dataset_valid.npz':pc.sha_file(mock/'dataset_valid.npz')}
 loaded,source=load_normalized('valid',mock,work,norm,expected_inputs);assert loaded['x'].shape==(2,2048)
 expect_error(lambda:load_normalized('valid',mock,work,norm,{'dataset_valid.npz':'bad'}))
 checks['feature_integrity_guard']=True
 data={};contexts={}
 for split,n in (('train',160),('valid',64)):
  y=np.tile([0,1],n//2).astype(np.int64);rng.shuffle(y)
  x=rng.normal(0,.2,(n,16)).astype(np.float32);x[:,0]=2*(2*y-1)+rng.normal(0,.15,n)
  pred=y.copy();bad=rng.choice(n,n//3,replace=False);pred[bad]=1-pred[bad]
  margin=(2*pred-1).astype(np.float32)*.30
  logits=np.stack((-margin,margin),1)
  data[split]=dict(ids=np.array(['%s%d$_$%d'%(split,i//4,i) for i in range(n)]),y=y,x=x)
  contexts[split]=dict(y=y,ht=rng.normal(0,.2,(n,8)).astype(np.float32),**{'logits_'+m:logits.copy() for m in MODES})
 # IDs use distinct synthetic names; no labels are supplied to the donor function.
 donors={s:info.cross_video(data[s]['ids'],17+i) for i,s in enumerate(('train','valid'))}
 warm=info.Probe('panns_embedding',16);optimizer=torch.optim.AdamW(warm.parameters(),lr=.01)
 for step in range(65):
  warm.train();optimizer.zero_grad();loss=nn.functional.cross_entropy(warm(torch.tensor(data['train']['x'])),torch.tensor(data['train']['y']))
  loss.backward();optimizer.step()
 warm_state=run.state(warm);warm_identity=dict(checkpoint_sha256=run.fingerprint(warm),synthetic=True)
 warmdir=work/'warm';warmdir.mkdir();warm_logits=info.predict(warm,data['valid']['x'],None,torch.device('cpu'))
 pc.atomic_npz(warmdir/'validation_predictions.npz',ids=data['valid']['ids'],labels=data['valid']['y'],
               pred_original=warm_logits.argmax(1),prob_original=info.prob(warm_logits))
 run.verify_warmstart(data['valid'],warmdir,warm_state,torch.device('cpu'))
 pc.atomic_npz(warmdir/'validation_predictions.npz',ids=data['valid']['ids'],labels=1-data['valid']['y'],
               pred_original=warm_logits.argmax(1),prob_original=info.prob(warm_logits))
 expect_error(lambda:run.verify_warmstart(data['valid'],warmdir,warm_state,torch.device('cpu')))
 checks['warmstart_readout_and_label_guard']=True
 args=SimpleNamespace(out=work/'runs',epochs=12,batch=32,device=torch.device('cpu'),audio_lr=.001,fusion_lr=.03,permutations=3,seeds=[13])
 args.out.mkdir();pc.dump_json(args.out/'pipeline_signature.json',dict(synthetic=True,test_evaluated=False))
 pc.atomic_npz(args.out/'validation_features.npz',ids=data['valid']['ids'],labels=data['valid']['y'],x=data['valid']['x'],ht=contexts['valid']['ht'],
               **{'logits_'+m:contexts['valid']['logits_'+m] for m in MODES})
 jobs=[run.fit(args,data,contexts,k,13,donors,warm_state,warm_identity) for k in KINDS]
 assert len({j['identity']['initialization_sha256'] for j in jobs})==1
 assert all(j['missing_audio_exact_identity'] for j in jobs)
 assert jobs[0]['modes']['TAV']['macro_f1']>.95
 assert jobs[0]['assigned_minus_permuted_pp']>20
 assert jobs[0]['modes']['TAV']['macro_f1']>jobs[2]['modes']['TAV']['macro_f1']
 checks['synthetic_known_audio_signal_learned_and_shuffle_sensitive']=True
 assert run.fit(args,data,contexts,KINDS[0],13,donors,warm_state,warm_identity)==jobs[0]
 checks['all_four_controls_shared_initialization_and_completed_resume']=True
 run.summarize(args,jobs)
 archive=args.out/'panns_fusion_review_20261028.zip'
 with zipfile.ZipFile(archive) as z:
  manifest=json.loads(z.read('panns_fusion_review/review_manifest.json'))
  for name,digest in manifest['files'].items():assert hashlib.sha256(z.read('panns_fusion_review/'+name)).hexdigest()==digest
  assert 'panns_fusion_review/validation_features.npz' in z.namelist()
 checks['review_archive_all_hashes_and_replay_features']=True
 checkpoint=args.out/KINDS[0]/'seed_13/best.pt'
 with checkpoint.open('ab') as f:f.write(b'changed')
 expect_error(lambda:run.fit(args,data,contexts,KINDS[0],13,donors,warm_state,warm_identity))
 checks['completed_checkpoint_mutation_rejected']=True
 # A supervised warmstart is kept in full_no_aux; only continuing auxiliary CE differs.
 assert jobs[3]['identity']['warmstart_sha256']==jobs[0]['identity']['warmstart_sha256']
 pc.dump_json(HERE/'checks/engineering_check.json',dict(status='pass',scope='synthetic_CPU_engineering_only',checks=checks,
  synthetic_results=[dict(kind=j['kind'],f1=j['modes']['TAV']['macro_f1'],gain=j['gain_over_parent_pp'],best_epoch=j['best_epoch']) for j in jobs],
  torch=torch.__version__,numpy=np.__version__,work=str(work),real_MOSEI_training=False,
  tested_source_sha256={f:pc.sha_file(HERE/f) for f in ('fusion_core.py','run_fusion.py','checks/check_fusion.py')},test_evaluated=False))
 print('FUSION_ENGINEERING_PASS',len(checks),flush=True)
if __name__=='__main__':main()