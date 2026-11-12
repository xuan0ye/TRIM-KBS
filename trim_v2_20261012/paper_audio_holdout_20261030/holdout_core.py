"""Fresh paper experts, disjoint video roles, and read-only held-out evaluation."""
from pathlib import Path
from types import SimpleNamespace
import gc,hashlib,json,os,random,sys,time
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
PARENT=Path(os.environ.get('PAPER_PARENT_DIR',HERE.parent/'paper_audio_repr_20261025')).resolve()
sys.path.insert(0,str(PARENT))
import paper_common as pc
import paper_train as pt
from paper_data import preflight_bert
from holdout_split import ROLES,identity,validate_plan
MODES=('T','TA','TV','TAV')
CONDITIONS=('gated_correct','gated_shuffled','gated_zero','constant_correct')

class AudioExpert(nn.Module):
 def __init__(self,input_dim=2048):
  super().__init__();self.net=nn.Sequential(nn.Linear(input_dim,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,2))
 def forward(self,x):return self.net(x)

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def prob(logits):
 a=np.exp(logits-logits.max(1,keepdims=True));return a[:,1]/a.sum(1)

def scores(y,logits):return pc.scores(y,logits.argmax(1),prob(logits))

def verify_release():
 m=pc.read_json(HERE/'bundle_manifest.json')
 for rel,digest in m['files'].items():
  if pc.sha_file(HERE/rel)!=digest:raise RuntimeError('Holdout source changed: '+rel)
 if pc.sha_file(PARENT/'bundle_manifest.json')!=m['parent_manifest_sha256']:raise RuntimeError('Paper source release changed')
 pc.verify_code();return pc.sha_file(HERE/'bundle_manifest.json')

def verify_inputs(run):
 done=pc.read_json(run/'paper_original/seed_42/complete.json')
 if done['status']!='complete' or done['test_evaluated']:raise RuntimeError('Unexpected parent data source')
 inputs=done['signature']['inputs']
 for rel,digest in inputs.items():
  path=PARENT/'bundle_manifest.json' if rel=='bundle_manifest_sha256' else run/rel
  if pc.sha_file(path)!=digest:raise RuntimeError('Source input changed: '+rel)
 plan=pc.read_json(run/'inventory.json')
 if plan['smoke'] or plan['data_sha256']!=pc.DATA_SHA:raise RuntimeError('Expected completed real MOSEI data')
 return inputs

def keep_stats(path,mean,std):
 if path.exists():
  old=pc.load_npz(path)
  if not np.array_equal(old['mean'],mean) or not np.array_equal(old['std'],std):raise RuntimeError('Expert-fit normalization changed')
 else:pc.atomic_npz(path,mean=mean,std=std)

def partition_training(train,raw_embedding,plan,normalization_path):
 validate_plan(plan,train['ids'],train['y'])
 if len(raw_embedding)!=len(train['y']) or not np.isfinite(raw_embedding).all():raise RuntimeError('Invalid raw embeddings')
 fit=np.asarray(plan['roles']['expert_train']['indices']);mean,std=pc.scale_fit(raw_embedding[fit])
 keep_stats(normalization_path,mean,std);data={}
 for role in ROLES:
  ix=np.asarray(plan['roles'][role]['indices']);paper={k:v[ix] for k,v in train.items()}
  x=np.clip(pc.scale_apply(raw_embedding[ix],(mean,std)),-8,8)
  data[role]=dict(paper=paper,x=x)
 return data

def prepare_training(args):
 inputs=verify_inputs(args.parent_run);plan=pc.read_json(HERE/'split_plan.json')
 if inputs['paper_train.npz']!=plan['parent_paper_train_sha256']:raise RuntimeError('Expected committed parent training file')
 train=pc.load_npz(args.parent_run/'paper_train.npz');embedding=pc.load_npz(args.parent_run/'embedding_train.npz')
 if not np.array_equal(train['ids'],embedding['ids']):raise RuntimeError('Raw PANNs feature IDs differ')
 if embedding['embedding'].shape!=(len(train['y']),2048):raise RuntimeError('Expected complete PANNs inputs')
 args.out.mkdir(parents=True,exist_ok=True)
 data=partition_training(train,embedding['embedding'],plan,args.out/'normalization.npz')
 for role in ('router_train','router_dev'):cross_video(data[role]['paper']['ids'],20261030)
 for role in ROLES:print('HOLDOUT_ROLE',role,'clips',plan['roles'][role]['n'],'videos',len(plan['roles'][role]['videos']),flush=True)
 pc.atomic_npz(args.out/'role_index.npz',ids=train['ids'],labels=train['y'],**{r:np.asarray(plan['roles'][r]['indices']) for r in ROLES})
 pc.record_json(args.out/'split_plan.json',plan)
 signature=dict(parent_inputs=inputs,split_plan_sha256=pc.sha_file(HERE/'split_plan.json'),
  normalization_sha256=pc.sha_file(args.out/'normalization.npz'),normalization_fit_role='expert_train',
  role_identity={r:plan['roles'][r]['ids_sha256'] for r in ROLES},test_evaluated=False)
 return data,signature

def prepare_external(args):
 # Invoked only after all experts and routers are complete and their selections are sealed.
 paper=pc.load_npz(args.parent_run/'paper_valid.npz');e=pc.load_npz(args.parent_run/'embedding_valid.npz')
 if not np.array_equal(paper['ids'],e['ids']):raise RuntimeError('External feature IDs differ')
 role_index=pc.load_npz(args.out/'role_index.npz')
 videos=lambda ids:{str(s).split('$_$')[0] for s in ids}
 if videos(paper['ids'])&videos(role_index['ids']):raise RuntimeError('External source video overlap')
 norm=pc.load_npz(args.out/'normalization.npz');x=np.clip(pc.scale_apply(e['embedding'],(norm['mean'],norm['std'])),-8,8)
 return dict(paper=paper,x=x)

def expert_parts(data):
 # Deliberately expose only these two roles to the unchanged paper trainer.
 return {'train':data['expert_train']['paper'],'valid':data['expert_dev']['paper']}

def check_completed(folder,signature):
 path=folder/'complete.json'
 if not path.exists():return None
 done=pc.read_json(path)
 if done['status']!='complete' or done['signature']!=signature:raise RuntimeError('Completed job signature changed: '+str(folder))
 for rel,digest in done['artifacts'].items():
  if pc.sha_file(folder/rel)!=digest:raise RuntimeError('Completed artifact changed: '+str(folder/rel))
 return done

def audio_logits(model,x,device,batch=256):
 model.eval();out=[]
 with torch.inference_mode():
  for start in range(0,len(x),batch):out.append(model(torch.as_tensor(x[start:start+batch],device=device)).cpu().numpy())
 return np.concatenate(out)

def criterion(y,device):
 counts=np.bincount(y,minlength=2)
 if counts.min()<1:raise RuntimeError('Both labels required')
 weights=np.sqrt(counts.sum()/(2*counts));weights=weights/weights.mean()
 return nn.CrossEntropyLoss(weight=torch.as_tensor(weights,dtype=torch.float32,device=device),label_smoothing=.05)

def train_audio(args,fit,dev,pipeline):
 folder=args.out/'audio_expert';folder.mkdir(parents=True,exist_ok=True)
 signature=dict(pipeline=pipeline,seed=args.seed,epochs=15,batch=64,lr=.001,weight_decay=.001,
  fit_role='expert_train',selection_role='expert_dev',initialization='new_random_classifier_no_old_sentiment_weights',test_evaluated=False)
 completed=check_completed(folder,signature);seed_all(args.seed)
 model=AudioExpert(fit['x'].shape[1]).to(args.device)
 initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
 if completed is None:
  ce=criterion(fit['paper']['y'],args.device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
  rng=np.random.default_rng(args.seed);best=-1.;best_epoch=0;history=[]
  for epoch in range(1,16):
   model.train();order=rng.permutation(len(fit['x']));loss_total=0.
   for start in range(0,len(order),64):
    ix=order[start:start+64];optimizer.zero_grad(set_to_none=True)
    logits=model(torch.as_tensor(fit['x'][ix],device=args.device));loss=ce(logits,torch.as_tensor(fit['paper']['y'][ix],device=args.device).long())
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite audio loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_total+=float(loss.detach().cpu())*len(ix)
   logits=audio_logits(model,dev['x'],args.device);m=scores(dev['paper']['y'],logits)
   if m['macro_f1']>best:best=m['macro_f1'];best_epoch=epoch;pt.save_torch(folder/'best.pt',pt.cpu_state(model))
   history.append(dict(epoch=epoch,train_loss=loss_total/len(fit['x']),expert_dev_macro_f1=m['macro_f1']))
   pc.dump_json(folder/'history.json',history)
   print('HOLDOUT_AUDIO_EPOCH',epoch,'expert_dev_f1',round(m['macro_f1'],6),flush=True)
  artifacts={r:pc.sha_file(folder/r) for r in ('best.pt','history.json')}
  pc.dump_json(folder/'complete.json',dict(status='complete',signature=signature,initialization_sha256=initial,best_epoch=best_epoch,
   best_expert_dev_macro_f1=best,artifacts=artifacts,test_evaluated=False))
 model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True),strict=True)
 return model.eval().requires_grad_(False)

def load_trim(args,bert,data):
 parts=expert_parts(data);folder=args.out/'trim_expert/paper_original'/f'seed_{args.seed}'
 model=pt.make_model(bert,parts['train']['audio'].shape[1],parts['train']['vision'].shape[1],pretrained=False)
 model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True),strict=True)
 return model.eval().requires_grad_(False).to(args.device)

def readouts(args,role,data,trim,audio,pipeline):
 folder=args.out/'readouts';folder.mkdir(parents=True,exist_ok=True);path=folder/(role+'.npz');meta=path.with_suffix('.json')
 signature=dict(pipeline=pipeline,role=role,ids_sha256=identity(data['paper']['ids'],data['paper']['y']),
  trim_sha256=pc.sha_file(args.out/'trim_expert/paper_original'/f'seed_{args.seed}/best.pt'),audio_sha256=pc.sha_file(args.out/'audio_expert/best.pt'))
 if path.exists() and meta.exists():
  old=pc.read_json(meta)
  if old['signature']!=signature or old['sha256']!=pc.sha_file(path):raise RuntimeError('Readout cache differs')
  return pc.load_npz(path)
 p=data['paper'];buffers={m:[] for m in MODES};trim.eval();audio.eval()
 with torch.inference_mode():
  for start in range(0,len(p['y']),args.batch):
   sl=slice(start,start+args.batch);tb=torch.as_tensor(p['text'][sl],device=args.device)
   ta=torch.as_tensor(p['audio'][sl],device=args.device);tv=torch.as_tensor(p['vision'][sl],device=args.device)
   hidden=pt.encode(trim,tb,ta,tv)
   for mode in MODES:
    mask=torch.tensor(pc.MASKS[mode],device=args.device).expand(len(tb),-1)
    buffers[mode].append(pt.fuse(trim,*hidden,mask).cpu().numpy())
 result=dict(ids=p['ids'],y=p['y'],audio_logits=audio_logits(audio,data['x'],args.device),
  zero_audio_logits=audio_logits(audio,np.zeros_like(data['x']),args.device),**{'logits_'+m:np.concatenate(v) for m,v in buffers.items()})
 pc.atomic_npz(path,**result);pc.dump_json(meta,dict(signature=signature,sha256=pc.sha_file(path)))
 print('HOLDOUT_READOUTS',role,len(p['y']),flush=True);return result

def cross_video(ids,seed):
 videos=np.array([str(s).split('$_$')[0] for s in ids]);names,counts=np.unique(videos,return_counts=True);n=len(ids);maximum=int(counts.max())
 if maximum>n//2:raise RuntimeError('Cross-video assignment impossible')
 rng=np.random.default_rng(seed);order=np.concatenate([rng.permutation(np.flatnonzero(videos==s)) for s in rng.permutation(names)])
 ix=np.empty(n,dtype=np.int64);ix[order]=np.roll(order,int(rng.integers(maximum,n-maximum+1)))
 assert len(np.unique(ix))==n and np.all(videos[ix]!=videos);return ix