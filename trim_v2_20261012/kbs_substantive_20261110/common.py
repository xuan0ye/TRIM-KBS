from pathlib import Path
import csv,hashlib,json,os,random,sys,time
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
REL={'MOSI':'MOSI/Processed/aligned_50.pkl','SIMS':'SIMS/Processed/unaligned_39.pkl','MOSEI':'MOSEI/Processed/aligned_50.pkl'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def dump(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');os.replace(tmp,p)
def csvout(path,rows):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def npz(path,**items):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 with tmp.open('wb')as f:np.savez_compressed(f,**items)
 os.replace(tmp,p)
def arrays(path):
 with np.load(path,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def verify_code():
 p=HERE/'bundle_manifest.json'
 if not p.exists():raise RuntimeError('Build the bundle manifest first.')
 for name,digest in read(p)['files'].items():
  if sha(HERE/name)!=digest:raise RuntimeError('Source changed; use a newly built package: '+name)
 return sha(p)
def plan():return read(HERE/'protocol.json')
def source_ids(ids):
 out=np.asarray([str(s).split('$_$')[0]for s in ids])
 if np.any(out==np.asarray(ids).astype(str)):raise ValueError('Source-video ID delimiter missing')
 return out
def metrics(y,margin):
 y=np.asarray(y,dtype=int);margin=np.asarray(margin)
 if len(y)==0:return {'n':0,'macro_f1':None,'accuracy':None,'two_classes':False}
 if not np.isfinite(margin).all():raise ValueError('Nonfinite prediction')
 q=(margin>0).astype(int);tn,fp,fn,tp=map(int,np.bincount(2*y+q,minlength=4))
 f1=.5*(2*tn/max(1,2*tn+fp+fn)+2*tp/max(1,2*tp+fp+fn))
 prob=1/(1+np.exp(-np.clip(margin,-80,80)))
 return dict(n=len(y),tn=tn,fp=fp,fn=fn,tp=tp,accuracy=float((q==y).mean()),macro_f1=float(f1)if len(np.unique(y))==2 else None,two_classes=len(np.unique(y))==2,
  nll=float(-(y*np.log(np.clip(prob,1e-12,1))+(1-y)*np.log(np.clip(1-prob,1e-12,1))).mean()),brier=float(((prob-y)**2).mean()))
def runtime(device='cuda:0',seed=0):
 import torch
 torch.set_num_threads(4);torch.set_default_dtype(torch.float32);torch.backends.cudnn.enabled=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.mha.set_fastpath_enabled(False)
 if str(device).startswith('cuda')and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; activate wjhenv.')
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 return torch.device(device)
def selection_data(out,dataset,role):
 if role not in ['fit','dev']:raise RuntimeError('Training may only load fit/dev.')
 p=Path(out)/'data'/dataset/f'{role}.npz';meta=read(p.parent/'prepared.json')
 if sha(p)!=meta['files'][p.name]:raise RuntimeError('Prepared data changed: '+str(p))
 return arrays(p)
def evaluation_data(out,dataset,role):
 if role not in ['audit','valid','test']:raise ValueError(role)
 seal=read(Path(out)/'training_seal.json')
 if seal['protocol_sha256']!=sha(HERE/'protocol.json'):raise RuntimeError('Evaluation protocol changed after training seal.')
 p=Path(out)/'data'/dataset/f'{role}.npz';meta=read(p.parent/'prepared.json')
 if sha(p)!=meta['files'][p.name]:raise RuntimeError('Sealed evaluation data changed: '+str(p))
 return arrays(p)