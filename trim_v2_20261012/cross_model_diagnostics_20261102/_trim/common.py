from pathlib import Path
import csv,hashlib,json,math,os
import numpy as np
import mmap_dataset_pickle as mm
DATASETS=("MOSI","SIMS","MOSEI")
VARIANTS=("full","no_modrop","text_only")
SEEDS=(13,21,42,57,88)
MASK_SEEDS=(13,21,42,57,73,88,101,123,202,340)
RATES=(.1,.3,.5,.7)
REL={"MOSI":"MOSI/Processed/aligned_50.pkl","SIMS":"SIMS/Processed/unaligned_39.pkl","MOSEI":"MOSEI/Processed/aligned_50.pkl"}
COMBOS={"text_only":(1,0,0),"text_audio":(1,1,0),"text_visual":(1,0,1),"full":(1,1,1)}
def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(8*1024*1024),b""):h.update(block)
 return h.hexdigest()
def dump_json(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8");os.replace(tmp,path)
def write_csv(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",encoding="utf-8-sig",newline="") as f:
  writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
def read_raw(root,dataset):
 path=Path(root)/REL[dataset]
 with path.open("rb") as f:return mm.load(f)
def prepare(split):
 original=np.asarray(split["classification_labels"]).reshape(-1)
 assert np.isin(original,[0,1,2]).all()
 keep=np.flatnonzero(original!=1)
 y=(original[keep]==2).astype(np.int8)
 ids=np.asarray([str(split["id"][i]) for i in keep])
 assert len(set(ids.tolist()))==len(ids)
 reg=np.asarray(split["regression_labels"]).reshape(-1)[keep]
 assert np.array_equal((reg>0).astype(np.int8),y) and np.all(reg!=0)
 text=np.asarray(split["text_bert"])[keep].astype(np.int64)
 pooled={}
 for key in ("audio","vision"):
  # Preserve historical float64-mean -> float32 conversion, without treating zeros as lengths.
  x=split[key];pooled[key]=np.stack([x[i].mean(0) for i in keep]).astype(np.float32)
  assert np.isfinite(pooled[key]).all(),key
 assert np.all((text[:,1]==0)|(text[:,1]==1))
 return dict(ids=ids,y=y,text=text,audio=pooled["audio"],vision=pooled["vision"],indices=keep)
def metric(y,p):
 y=np.asarray(y).astype(int);p=np.asarray(p).astype(int)
 assert len(y)>0 and y.shape==p.shape and np.isin(p,[0,1]).all()
 tn,fp,fn,tp=map(int,np.bincount(2*y+p,minlength=4))
 f0=2*tn/max(1,2*tn+fp+fn);f1=2*tp/max(1,2*tp+fp+fn)
 den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
 return dict(n_test=len(y),tn=tn,fp=fp,fn=fn,tp=tp,accuracy=(tn+tp)/len(y),macro_f1=(f0+f1)/2,balanced_accuracy=(tn/max(1,tn+fp)+tp/max(1,tp+fn))/2,mcc=(tp*tn-fp*fn)/den if den else 0.,weighted_f1=((tn+fp)*f0+(tp+fn)*f1)/len(y))
def draw_masks(n,p,seed):
 rng=np.random.default_rng(seed*1000003+int(round(p*1000))*7919)
 z=rng.random((n,2))>=p
 return z[:,0].astype(np.int8)+2*z[:,1].astype(np.int8)
def expected_index():
 value=json.loads((Path(__file__).parent/"expected_results.json").read_text(encoding="utf8"))
 return {(r["dataset"],r["variant"],int(r["seed"])):r for r in value["ablations"]}
