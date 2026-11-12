"""Audio-only information probes. Frozen TRIM parents and test data are never trained/evaluated."""
from pathlib import Path
import csv,hashlib,json,os,sys
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
PARENT=Path(os.environ.get('PAPER_PARENT_DIR',HERE.parent/'paper_audio_repr_20261025')).resolve()
sys.path.insert(0,str(PARENT))
import paper_common as pc
from common import read_raw
RECIPES=('length_only','stored_mean','observed_mean','observed_moments','panns_embedding',
         'temporal_cnn','frame_shuffled_cnn','label_shuffled_cnn')
CNN_RECIPES=set(RECIPES[-3:])
SEEDS=(13,21,42)
SPLITS=('train','valid')

def runtime(device='cuda:0',threads=4):
 torch.set_num_threads(threads);torch.set_default_dtype(torch.float32)
 torch.backends.cudnn.enabled=False;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 return torch.device(device)
def verify_release():
 m=pc.read_json(HERE/'bundle_manifest.json')
 for rel,digest in m['files'].items():
  if pc.sha_file(HERE/rel)!=digest:raise RuntimeError(f'Information-probe source changed: {rel}')
 pc.verify_code()
 if pc.sha_file(PARENT/'bundle_manifest.json')!=m['parent_manifest_sha256']:raise RuntimeError('Parent release differs')
 return pc.sha_file(HERE/'bundle_manifest.json')
def cross_video(ids,seed):
 v=np.array([str(x).split('$_$')[0] for x in ids]);names,counts=np.unique(v,return_counts=True);n=len(ids)
 maximum=int(counts.max())
 if maximum>n//2:raise ValueError('No cross-video bijection possible')
 rng=np.random.default_rng(seed)
 order=np.concatenate([rng.permutation(np.flatnonzero(v==name)) for name in rng.permutation(names)])
 ix=np.empty(n,dtype=np.int64);ix[order]=np.roll(order,int(rng.integers(maximum,n-maximum+1)))
 assert np.all(v[ix]!=v) and len(np.unique(ix))==n
 return ix
def frame_indices(ids,mask,seed):
 out=np.tile(np.arange(mask.shape[1]),(len(ids),1))
 for i,sid in enumerate(ids):
  positions=np.flatnonzero(mask[i]);n=len(positions)
  if n>1:
   sub_seed=int.from_bytes(hashlib.sha256((str(seed)+':'+str(sid)).encode()).digest()[:8],'little')
   out[i,positions]=positions[pc.derangement(n,sub_seed)]
 return out
def frame_permute(x,mask,ids,seed):
 ix=frame_indices(ids,mask,seed)
 return x[np.arange(len(x))[:,None],ix]
def compact_audio(audio):
 active=np.any(audio!=0,axis=-1);n,t,d=audio.shape
 seq=np.zeros((n,t,d),dtype=np.float32);mask=np.zeros((n,t),dtype=bool)
 for i in range(n):
  rows=audio[i,active[i]];seq[i,:len(rows)]=rows;mask[i,:len(rows)]=True
 return seq,mask,active
def observed_summary(audio,active):
 n=active.sum(1).clip(min=1).astype(np.float64)
 mean=(audio*active[:,:,None]).sum(1)/n[:,None]
 second=audio**2
 var=(second*active[:,:,None]).sum(1)/n[:,None]-mean**2
 std=np.sqrt(np.maximum(var,0))
 return mean.astype(np.float32),np.concatenate((mean,std),axis=1).astype(np.float32)
def normalize_pair(train,valid,stats,prefix):
 fit=pc.scale_fit(train);stats[prefix+'_mean']=fit[0];stats[prefix+'_std']=fit[1]
 return tuple(np.clip(pc.scale_apply(x,fit),-8,8) for x in (train,valid))
def prepare_arrays(parts,audios,embeds,out,metadata):
 results={};raw={};stats={};quality=[]
 for s in SPLITS:
  audio=np.asarray(audios[s],dtype=np.float64)
  if not np.isfinite(audio).all() or audio.shape[0]!=len(parts[s]['ids']):raise RuntimeError('Invalid aligned audio')
  seq,mask,active=compact_audio(audio)
  if not mask.any(1).all():raise RuntimeError('All-zero audio clip: refuse automatic removal')
  observed,moments=observed_summary(audio,active)
  stored=audio.mean(1).astype(np.float32)
  if not np.allclose(stored,parts[s]['audio'],rtol=1e-6,atol=1e-6):raise RuntimeError('Stored-frame mean does not match parent')
  results[s]=dict(ids=parts[s]['ids'],y=parts[s]['y'],seq=seq,mask=mask)
  raw[s]=dict(length_only=active.sum(1).astype(np.float32)[:,None],stored_mean=stored,
              observed_mean=observed,observed_moments=moments,panns_embedding=embeds[s].astype(np.float32))
  text=parts[s]['text'];textmask=(text[:,1]>0)&(text[:,0]!=101)&(text[:,0]!=102)
  mismatched=np.any(active!=textmask,axis=1)
  quality.append(dict(split=s,n=len(audio),all_zero_clips=0,first_audio_row_always_zero=bool((~active[:,0]).all()),
   nonzero_row_count_quantiles=np.quantile(active.sum(1),[0,.1,.5,.9,1]).tolist(),
   audio_text_mask_disagreement_count=int(mismatched.sum()),
   disagreements=[dict(id=str(parts[s]['ids'][i]),audio_positions=np.flatnonzero(active[i]).tolist(),
      text_nonspecial_positions=np.flatnonzero(textmask[i]).tolist()) for i in np.flatnonzero(mismatched)],
   mask_policy='audio nonzero-row proxy; compact in original order; text mask is audited only, never used as model input'))
 for recipe in RECIPES[:5]:
  a,b=normalize_pair(raw['train'][recipe],raw['valid'][recipe],stats,recipe)
  results['train'][recipe]=a;results['valid'][recipe]=b
 frame_train=results['train']['seq'][results['train']['mask']]
 fit=pc.scale_fit(frame_train);stats['frame_mean']=fit[0];stats['frame_std']=fit[1]
 for s in SPLITS:
  norm=np.clip(pc.scale_apply(results[s]['seq'],fit),-8,8)
  results[s]['seq']=np.where(results[s]['mask'][:,:,None],norm,0).astype(np.float32)
  pc.atomic_npz(out/f'dataset_{s}.npz',**results[s])
 permutation=pc.derangement(len(results['train']['y']),20261028)
 pc.atomic_npz(out/'normalization.npz',**stats)
 pc.atomic_npz(out/'data_index.npz',train_ids=results['train']['ids'],valid_ids=results['valid']['ids'],
               train_y=results['train']['y'],valid_y=results['valid']['y'],label_shuffle_train_donors=permutation,
               train_lengths=results['train']['mask'].sum(1),valid_lengths=results['valid']['mask'].sum(1))
 pc.dump_json(out/'audio_quality.json',dict(quality=quality,metadata=metadata,test_evaluated=False))
 return results,permutation
def prepare(root,run,out):
 done=pc.read_json(run/'paper_original/seed_42/complete.json')
 inputs=done['signature']['inputs']
 required=('inventory.json','paper_train.npz','paper_valid.npz','embedding_train.npz','embedding_valid.npz')
 for rel in required:
  if pc.sha_file(run/rel)!=inputs[rel]:raise RuntimeError(f'Parent input changed: {rel}')
 plan=pc.read_json(run/'inventory.json')
 if plan['smoke'] or plan['test_evaluated'] or set(plan['splits'])!=set(SPLITS):raise RuntimeError('Expected real train/valid pilot')
 data_path=root/'MSA Datasets/MOSEI/Processed/aligned_50.pkl'
 print('AUDIO_INFORMATION_VERIFY_ALIGNED_DATA',data_path,flush=True)
 digest=pc.sha_file(data_path)
 if digest!=pc.DATA_SHA or plan['data_sha256']!=digest:raise RuntimeError('Aligned data differs')
 from paper_data import load_parts
 parts,embeds=load_parts(run)
 raw_data=read_raw(root/'MSA Datasets','MOSEI');audios={}
 for s in SPLITS:
  split=raw_data[s];lookup={str(sid):i for i,sid in enumerate(split['id'])}
  ix=np.array([lookup[str(sid)] for sid in parts[s]['ids']])
  labels=np.asarray(split['classification_labels']).reshape(-1)[ix]
  if not np.array_equal((labels==2).astype(np.int8),parts[s]['y']) or np.any(labels==1):raise RuntimeError('Labels differ')
  audios[s]=np.stack([split['audio'][i] for i in ix])
 if set(parts['train']['ids']).intersection(parts['valid']['ids']):raise RuntimeError('Split overlap')
 signature=dict(data_sha256=digest,parent_inputs={f:inputs[f] for f in required},mask='nonzero_audio_row_proxy',
                normalization='training_only_clip8',test_evaluated=False)
 results,permutation=prepare_arrays(parts,audios,embeds,out,signature)
 pc.dump_json(out/'data_signature.json',signature)
 return results,permutation,signature

class Probe(nn.Module):
 def __init__(self,recipe,input_dim):
  super().__init__();self.is_sequence=recipe in CNN_RECIPES
  if self.is_sequence:
   self.project=nn.Sequential(nn.Linear(input_dim,32),nn.LayerNorm(32),nn.GELU())
   self.conv1=nn.Conv1d(32,32,3,padding=1);self.conv2=nn.Conv1d(32,32,3,padding=1)
   self.dropout=nn.Dropout(.1);self.classifier=nn.Linear(64,2)
  else:self.net=nn.Sequential(nn.Linear(input_dim,32),nn.LayerNorm(32),nn.GELU(),nn.Dropout(.1),nn.Linear(32,2))
 def forward(self,x,mask=None):
  if not self.is_sequence:return self.net(x)
  m=mask[:,None,:]
  h=self.project(x).transpose(1,2)*m
  h=torch.nn.functional.gelu(self.conv1(h))*m
  h=self.dropout(h)
  h=torch.nn.functional.gelu(self.conv2(h))*m
  mean=h.sum(2)/m.sum(2).clamp(min=1)
  maximum=h.masked_fill(~m,-1e4).max(2).values
  maximum=torch.where(m.any(2),maximum,torch.zeros_like(maximum))
  return self.classifier(torch.cat((mean,maximum),1))

def representation(data,recipe):
 x=data['seq'] if recipe in CNN_RECIPES else data[recipe]
 mask=data['mask'] if recipe in CNN_RECIPES else None
 if recipe=='frame_shuffled_cnn':x=frame_permute(x,mask,data['ids'],20261028)
 return x,mask
def predict(model,x,mask,device,batch=256):
 model.eval();outputs=[]
 with torch.inference_mode():
  for start in range(0,len(x),batch):
   sl=slice(start,start+batch)
   outputs.append(model(torch.as_tensor(x[sl],device=device),
     None if mask is None else torch.as_tensor(mask[sl],device=device)).cpu().numpy())
 return np.concatenate(outputs)
def prob(logits):
 v=logits-logits.max(1,keepdims=True);ex=np.exp(v);return ex[:,1]/ex.sum(1)
def metrics(y,logits):
 return pc.scores(y,logits.argmax(1),prob(logits))