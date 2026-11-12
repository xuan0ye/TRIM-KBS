"""Shared deterministic interventions and metrics; no fitting or selection."""
from pathlib import Path
import csv,hashlib,json,math,os
import numpy as np
HERE=Path(__file__).resolve().parent
METHODS=('TRIM','MulT','LNLN');SEEDS=(13,21,42,57,88);DATASETS=('MOSI','SIMS')
FIXED={'full':(1,1,1),'text_only':(1,0,0),'text_audio':(1,1,0),'text_visual':(1,0,1),
       'zero_audio_present':(1,1,1),'zero_vision_present':(1,1,1),'zero_av_present':(1,1,1)}
def sha_file(p,algorithm='sha256'):
 h=hashlib.new(algorithm)
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,obj):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');os.replace(tmp,p)
def csvout(p,rows):
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def cross_video(ids,seed):
 videos=np.asarray([str(s).split('$_$')[0]for s in ids]);names,counts=np.unique(videos,return_counts=True);n=len(ids)
 largest=int(counts.max())
 if largest>n//2:raise ValueError('No cross-video bijection exists')
 rng=np.random.default_rng(seed);order=np.concatenate([rng.permutation(np.flatnonzero(videos==v))for v in rng.permutation(names)])
 donor=np.empty(n,dtype=np.int64);donor[order]=np.roll(order,int(rng.integers(largest,n-largest+1)))
 assert len(np.unique(donor))==n and np.all(videos[donor]!=videos)
 return donor
def sequence_order(x,ids,seed):
 # Nonzero-row positions are an observation proxy. Keep all zero rows in place.
 out=x.copy();changed=0
 for i,sid in enumerate(ids):
  pos=np.flatnonzero(np.any(x[i]!=0,axis=-1));n=len(pos)
  if n<2:continue
  rng=np.random.default_rng(int.from_bytes(hashlib.sha256((str(seed)+'|'+str(sid)).encode()).digest()[:8],'little'))
  permutation=rng.permutation(n)
  if np.array_equal(permutation,np.arange(n)):permutation=np.roll(permutation,1)
  out[i,pos]=x[i,pos[permutation]];changed+=int(not np.array_equal(out[i],x[i]))
 assert np.array_equal(np.any(out!=0,axis=-1),np.any(x!=0,axis=-1))
 return out,changed

def cases(ids,repetitions=20,temporal=5):
 definitions=[dict(name=k,kind=k,rep=-1,mask=list(v))for k,v in FIXED.items()];donors={}
 for rep in range(repetitions):
  da=cross_video(ids,20261102+rep*101);dv=cross_video(ids,20261103+rep*101)
  for family in ['audio_cross_video','vision_cross_video','paired_av_cross_video']:
   name=f'{family}_{rep:02}';definitions.append(dict(name=name,kind=family,rep=rep,mask=[1,1,1]))
   # Joint A/V donor is shared so their original mutual relation is preserved.
   identity=np.arange(len(ids),dtype=np.int64)
   donors[name]=(da,identity) if family=='audio_cross_video' else ((identity,dv) if family=='vision_cross_video' else (da,da))
 for rep in range(temporal):
  for kind in ['audio_order','vision_order','av_order']:
   definitions.append(dict(name=f'{kind}_{rep:02}',kind=kind,rep=rep,mask=[1,1,1]))
 return definitions,donors

def metrics(y,p):
 tn,fp,fn,tp=map(int,np.bincount(2*y.astype(int)+p.astype(int),minlength=4));n=len(y)
 den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
 return dict(n=n,tn=tn,fp=fp,fn=fn,tp=tp,accuracy=(tn+tp)/n,
  macro_f1=(2*tn/max(1,2*tn+fp+fn)+2*tp/max(1,2*tp+fp+fn))/2,
  balanced_accuracy=(tn/max(1,tn+fp)+tp/max(1,tp+fn))/2,mcc=(tp*tn-fp*fn)/den if den else 0.)
def compare(y,base,p,base_score,score):
 m=metrics(y,p);b=metrics(y,base);changed=base!=p
 return dict(**m,f1_delta_pp=100*(m['macro_f1']-b['macro_f1']),changed=int(changed.sum()),changed_fraction=float(changed.mean()),
  corrected=int(((base!=y)&(p==y)).sum()),broken=int(((base==y)&(p!=y)).sum()),
  mean_abs_native_score_delta=float(np.abs(score-base_score).mean()),max_abs_native_score_delta=float(np.abs(score-base_score).max()))
def verify_bundle():
 m=json.loads((HERE/'bundle_manifest.json').read_text('utf8'))
 for rel,digest in m['files'].items():
  if sha_file(HERE/rel)!=digest:raise RuntimeError('Bundle source/input hash mismatch: '+rel)
 return sha_file(HERE/'bundle_manifest.json')
def checkpoint(root,dataset,method,seed):
 index=json.loads((HERE/'checkpoint_manifest.json').read_text('utf8'))
 item=next((r for r in index if r['dataset']==dataset and r['method']==method and r['seed']==seed),None)
 candidates=[]
 if item:candidates.append(root/item['path'])
 if method=='TRIM':candidates.append(root/f'results_ccfb_2_4_full/train/{dataset}/full/seed_{seed}/{dataset}_e2e_best.pt')
 if method=='LNLN':
  suffix=f'audit_20260929/sota_native_runs/{dataset}/LNLN/seed_{seed}/best_val_macro_f1_seed_{seed}.pth'
  candidates.extend([root/'sota_repro_20261001'/suffix,root/suffix])
 if method=='MulT':
  suffix=f'{dataset}/MulT/seed_{seed}/{dataset}_cormult_repro_seed{seed}.pt'
  candidates.extend([root/'results_autodl/cormult'/suffix,root/'results_cormult_repro'/suffix])
 candidates=list(dict.fromkeys(p.resolve()for p in candidates));present=[p for p in candidates if p.is_file()]
 if not present:raise FileNotFoundError('Missing '+method+' checkpoint. Tried: '+', '.join(map(str,candidates)))
 found=[(p,sha_file(p))for p in present]
 if item:
  found=[r for r in found if r[1]==item['sha256']]
  if not found:raise RuntimeError('Checkpoint differs from the audited identity: '+str(present))
 elif len({s for _,s in found})>1:raise RuntimeError('Multiple distinct checkpoint candidates; resolve explicitly: '+str(present))
 return found[0][0],found[0][1],item is not None