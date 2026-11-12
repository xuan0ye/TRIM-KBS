"""Complete PANNs input with supervised audio readout and a frozen-TRIM residual gate."""
from pathlib import Path
import gc,hashlib,json,os,sys
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
PARENT=Path(os.environ.get('PAPER_PARENT_DIR',HERE.parent/'paper_audio_repr_20261025')).resolve()
CAUSAL=Path(os.environ.get('PAPER_CAUSAL_DIR',HERE.parent/'paper_audio_causal_20261026')).resolve()
INFO=Path(os.environ.get('PAPER_AUDIO_INFO_DIR',HERE.parent/'paper_audio_information_20261028')).resolve()
for path in (PARENT,CAUSAL,INFO):sys.path.insert(0,str(path))
import paper_common as pc
import causal_common as causal
import information_core as info
MODES=('T','TA','TV','TAV')
KINDS=('full_supervised','shuffled_supervised','zero_supervised','full_no_aux')
SEEDS=(13,21,42)

def verify_release():
 manifest=pc.read_json(HERE/'bundle_manifest.json')
 for rel,digest in manifest['files'].items():
  if pc.sha_file(HERE/rel)!=digest:raise RuntimeError(f'Fusion source changed: {rel}')
 for name,path in (('parent',PARENT),('causal',CAUSAL),('information',INFO)):
  if pc.sha_file(path/'bundle_manifest.json')!=manifest['dependencies'][name]:
   raise RuntimeError(f'Expected frozen {name} release')
 pc.verify_code();causal.verify_release();info.verify_release()
 return pc.sha_file(HERE/'bundle_manifest.json')

def load_normalized(split,info_run,parent_run,norm,expected_inputs):
 path=info_run/f'dataset_{split}.npz';source={}
 if path.exists():
  if pc.sha_file(path)!=expected_inputs[f'dataset_{split}.npz']:raise RuntimeError(f'Information dataset changed: {split}')
  with np.load(path,allow_pickle=False) as f:
   result=dict(ids=f['ids'],y=f['y'],x=f['panns_embedding'])
  source=dict(method='verified_information_dataset',sha256=pc.sha_file(path))
 else:
  done=pc.read_json(parent_run/'paper_original/seed_42/complete.json')
  for rel in (f'paper_{split}.npz',f'embedding_{split}.npz'):
   if pc.sha_file(parent_run/rel)!=done['signature']['inputs'][rel]:raise RuntimeError('Parent features changed')
  p=pc.load_npz(parent_run/f'paper_{split}.npz');e=pc.load_npz(parent_run/f'embedding_{split}.npz')
  if not np.array_equal(p['ids'],e['ids']):raise RuntimeError('Parent embedding IDs differ')
  x=np.clip(pc.scale_apply(e['embedding'],(norm['panns_embedding_mean'],norm['panns_embedding_std'])),-8,8)
  result=dict(ids=p['ids'],y=p['y'],x=x)
  source=dict(method='reconstructed_from_verified_parent_and_training_statistics',embedding_sha256=pc.sha_file(parent_run/f'embedding_{split}.npz'))
 if result['x'].shape!=(len(result['y']),2048) or not np.isfinite(result['x']).all():raise RuntimeError('Expected complete finite 2048D embeddings')
 if np.max(np.abs(result['x']))>8.00001:raise RuntimeError('Input clipping differs from information probe')
 source['values_sha256']=hashlib.sha256(result['x'].tobytes()).hexdigest()
 return result,source

def read_cache(path,expected):
 meta=path.with_suffix('.json')
 if path.exists() and meta.exists():
  record=pc.read_json(meta)
  if record['signature']!=expected or pc.sha_file(path)!=record['sha256']:raise RuntimeError(f'Frozen parent cache changed: {path}')
  return pc.load_npz(path),dict(path=str(path),sha256=record['sha256'])
 return None,None

def prepare(args):
 print('PANNS_FUSION_PREFLIGHT verify_parent_artifacts',flush=True)
 inputs=causal.verify_parent(args.parent_run)
 info_pipeline=pc.read_json(args.info_run/'pipeline_signature.json')
 if info_pipeline['release']!=pc.sha_file(INFO/'bundle_manifest.json'):raise RuntimeError('Information run release differs')
 if info_pipeline['data']['parent_inputs']['inventory.json']!=inputs['inventory.json']:raise RuntimeError('Parent sample definitions differ')
 for rel in ('normalization.npz','data_index.npz'):
  if pc.sha_file(args.info_run/rel)!=info_pipeline['data_files'][rel]:raise RuntimeError(f'Information statistics changed: {rel}')
 norm=pc.load_npz(args.info_run/'normalization.npz');index=pc.load_npz(args.info_run/'data_index.npz')
 data={};sources={}
 for split in ('train','valid'):
  print('PANNS_FUSION_PREFLIGHT load_full_embedding',split,flush=True)
  data[split],sources[split]=load_normalized(split,args.info_run,args.parent_run,norm,info_pipeline['data_files'])
  if not np.array_equal(data[split]['ids'],index[split+'_ids']) or not np.array_equal(data[split]['y'],index[split+'_y']):
   raise RuntimeError('Information input IDs/labels differ')
 train_videos={str(x).split('$_$')[0] for x in data['train']['ids']}
 valid_videos={str(x).split('$_$')[0] for x in data['valid']['ids']}
 if train_videos&valid_videos:raise RuntimeError('Cross-split source video overlap')
 contexts={};context_sources={};backbone=None;parts=None
 for split in ('train','valid'):
  print('PANNS_FUSION_PREFLIGHT frozen_context',split,flush=True)
  expected=causal.feature_signature(args.parent_run,'paper_original',split)
  context,source=read_cache(args.causal_run/'cache'/f'paper_original_{split}.npz',expected)
  if context is None:context,source=read_cache(args.out/'cache'/f'paper_original_{split}.npz',expected)
  if context is None:
   if parts is None:
    from paper_data import load_parts
    parts,_=load_parts(args.parent_run)
   if backbone is None:
    backbone=causal.load_backbone(args.root,args.parent_run,'paper_original',parts['train']['audio'].shape[1],
                                  parts['train']['vision'].shape[1],args.device)
   path=args.out/'cache'/f'paper_original_{split}.npz'
   context=causal.encode_cache(backbone,parts[split],path,expected,args.device)
   source=dict(path=str(path),sha256=pc.sha_file(path))
  if not np.array_equal(context['ids'],data[split]['ids']) or not np.array_equal(context['y'],data[split]['y']):raise RuntimeError('Parent cache ID/label mismatch')
  if context['ht'].shape!=(len(context['y']),256) or not np.isfinite(context['ht']).all():raise RuntimeError('Parent text projection differs')
  for mode in MODES:
   if context['logits_'+mode].shape!=(len(context['y']),2) or not np.isfinite(context['logits_'+mode]).all():
    raise RuntimeError('Invalid parent logits')
  contexts[split]=context;context_sources[split]=source
 if backbone is not None:del backbone;gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()
 original=pc.load_npz(args.parent_run/'paper_original/seed_42/validation_predictions.npz')
 for mode in MODES:
  logits=contexts['valid']['logits_'+mode]
  if not np.array_equal(logits.argmax(1),original['pred_'+mode]) or np.max(np.abs(info.prob(logits)-original['prob_'+mode]))>1e-5:
   raise RuntimeError('Parent checkpoint replay mismatch')
 signature=dict(parent_inputs=inputs,info_pipeline_sha256=pc.sha_file(args.info_run/'pipeline_signature.json'),
                normalization_sha256=pc.sha_file(args.info_run/'normalization.npz'),sources=sources,parent_context=context_sources,
                baseline_parent_seed=42,train_n=len(data['train']['y']),valid_n=len(data['valid']['y']),test_evaluated=False)
 pc.record_json(args.out/'data_signature.json',signature)
 pc.atomic_npz(args.out/'validation_features.npz',ids=data['valid']['ids'],labels=data['valid']['y'],
               x=data['valid']['x'],ht=contexts['valid']['ht'],
               **{'logits_'+m:contexts['valid']['logits_'+m] for m in MODES})
 donors={s:info.cross_video(data[s]['ids'],20261028+i) for i,s in enumerate(('train','valid'))}
 pc.atomic_npz(args.out/'assignments.npz',train_ids=data['train']['ids'],valid_ids=data['valid']['ids'],
               train_donors=donors['train'],valid_donors=donors['valid'])
 print('PANNS_FUSION_DATA_READY','train',len(data['train']['y']),'valid',len(data['valid']['y']),'test_evaluated=False',flush=True)
 return data,contexts,signature,donors

class SupervisedFusion(nn.Module):
 """All parent computations are detached; zero strength initializes an exact identity."""
 def __init__(self,input_dim=2048,text_dim=256):
  super().__init__();self.audio=info.Probe('panns_embedding',input_dim)
  self.gate=nn.Sequential(nn.Linear(text_dim+4,32),nn.GELU(),nn.Linear(32,1))
  nn.init.constant_(self.gate[-1].bias,-2.)
  self.strength=nn.Parameter(torch.zeros(()))
 def forward(self,base_logits,ht,x,audio_present):
  base=base_logits.detach();a=self.audio(x.detach())
  context=torch.cat((ht.detach(),base.softmax(-1).detach(),a.softmax(-1).detach()),dim=-1)
  gate=torch.sigmoid(self.gate(context).squeeze(-1))
  margin=.5*(a[:,1]-a[:,0])
  delta=torch.tanh(self.strength)*gate*2*torch.tanh(margin/2)
  delta=delta*audio_present.to(delta.dtype)
  return base+torch.stack((-delta,delta),dim=-1),a,delta,gate

def audio_source(info_run,seed):
 folder=info_run/'panns_embedding'/f'seed_{seed}';done=pc.read_json(folder/'complete.json')
 if done['status']!='complete' or done['signature']['recipe']!='panns_embedding' or done['signature']['seed']!=seed:
  raise RuntimeError('Audio warmstart differs')
 for rel,digest in done['artifacts'].items():
  if pc.sha_file(folder/rel)!=digest:raise RuntimeError(f'Audio warmstart artifact changed: {rel}')
 return folder,done

def predict(model,z,x,mode,device,batch=256):
 model.eval();logits=[];audio=[];delta=[];gates=[]
 with torch.inference_mode():
  for start in range(0,len(z['y']),batch):
   sl=slice(start,start+batch);n=len(z['y'][sl])
   output,a,d,g=model(torch.as_tensor(z['logits_'+mode][sl],device=device),torch.as_tensor(z['ht'][sl],device=device),
                     torch.as_tensor(x[sl],device=device),torch.full((n,),causal.BITS[mode][1],device=device))
   logits.append(output.cpu().numpy());audio.append(a.cpu().numpy());delta.append(d.cpu().numpy());gates.append(g.cpu().numpy())
 return np.concatenate(logits),np.concatenate(audio),np.concatenate(delta),np.concatenate(gates)