"""Sequence inputs. Native follows the same released-prefix convention as the LNLN loader."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from common import read_raw,sha_file,dump_json
import mmap_dataset_pickle as mm
def data_path(root,d,view):
 name="unaligned_39.pkl" if d=="SIMS" else ("unaligned_50.pkl" if view=="native" else "aligned_50.pkl")
 return Path(root)/"MSA Datasets"/d/"Processed"/name
def read_dataset(root,d,view):
 with data_path(root,d,view).open("rb") as f:return mm.load(f)
class SequenceDataset(Dataset):
 def __init__(self,raw,d,split,view,limit=None):
  self.raw=raw[split];self.dataset=d;self.split=split;self.view=view
  labels=np.asarray(self.raw["classification_labels"]).reshape(-1);assert np.isin(labels,[0,1,2]).all()
  self.indices=np.flatnonzero(labels!=1);self.y=(labels[self.indices]==2).astype(np.int64)
  reg=np.asarray(self.raw["regression_labels"]).reshape(-1)[self.indices]
  assert np.all(reg!=0) and np.array_equal(reg>0,self.y==1)
  self.ids=np.asarray([str(self.raw["id"][i]) for i in self.indices]);assert len(set(self.ids.tolist()))==len(self.ids)
  self.a_dim=self.raw["audio"].shape[-1];self.v_dim=self.raw["vision"].shape[-1]
  self.support="released_prefix_lengths" if d=="SIMS" or view=="native" else "aligned_text_without_cls_sep"
  text=np.asarray(self.raw["text_bert"])[self.indices];mask=text[:,1].astype(bool);lengths=mask.sum(1)
  assert lengths.min()>=2 and np.array_equal(mask,np.arange(mask.shape[1])[None]<lengths[:,None])
  if self.support=="released_prefix_lengths":
   for key in ("audio","vision"):
    ll=np.asarray(self.raw[key+"_lengths"]).reshape(-1)
    assert len(ll)==len(labels) and np.all(ll==ll.astype(np.int64)) and np.all((ll>=0)&(ll<=self.raw[key].shape[1]))
  if limit is not None:
   rng=np.random.default_rng(42);half=limit//2
   pick=np.r_[rng.choice(np.flatnonzero(self.y==0),half,replace=False),rng.choice(np.flatnonzero(self.y==1),limit-half,replace=False)]
   self.indices=self.indices[pick];self.y=self.y[pick];self.ids=self.ids[pick]
 def __len__(self):return len(self.indices)
 def support_mask(self,original,key):
  t=self.raw[key].shape[1]
  if self.support=="released_prefix_lengths":return np.arange(t)<int(self.raw[key+"_lengths"][original])
  mask=np.asarray(self.raw["text_bert"])[original,1].astype(bool).copy()
  assert len(mask)==t
  last=int(mask.sum())-1;mask[0]=False;mask[last]=False
  # The aligned representation has verified zeros at CLS/SEP/padding. Never infer support by zero values.
  assert np.count_nonzero(self.raw[key][original][~mask])==0,(self.dataset,self.split,key,original,"aligned support violation")
  return mask
 def __getitem__(self,index):
  original=int(self.indices[index]);result=dict(text=torch.tensor(np.asarray(self.raw["text_bert"])[original],dtype=torch.long),y=torch.tensor(int(self.y[index]),dtype=torch.long),index=torch.tensor(index))
  for key,name in (("audio","audio"),("vision","visual")):
   x=np.asarray(self.raw[key][original],dtype=np.float32).copy();valid=self.support_mask(original,key)
   # This is the explicit upstream LNLN audio sanitization rule; other nonfinite observations fail loudly.
   if key=="audio":x[np.isneginf(x)]=0.
   if not np.isfinite(x[valid]).all():raise ValueError(f"Nonfinite observed feature: {self.dataset}/{self.split}/{key}/{original}")
   x[~valid]=0.
   result[name]=torch.from_numpy(x);result[name+"_mask"]=torch.from_numpy(valid)
  return result
def collate(items):
 result={k:torch.stack([x[k] for x in items]) for k in items[0]}
 text_n=max(2,int(result["text"][:,1].sum(1).max()));result["text"]=result["text"][:,:,:text_n]
 for key in ("audio","visual"):
  mask=result[key+"_mask"];indices=torch.arange(mask.shape[1])[None].expand_as(mask)
  end=max(1,int(torch.where(mask,indices+1,torch.zeros_like(indices)).max()))
  result[key]=result[key][:,:end];result[key+"_mask"]=mask[:,:end]
 return result
def make_loader(ds,batch,shuffle=False,generator=None):
 return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,collate_fn=collate,generator=generator)
def to_device(batch,device):return {k:v.to(device,non_blocking=False) for k,v in batch.items()}
def train_statistics(ds):
 result={}
 for raw_key,name in (("audio","audio"),("vision","vision")):
  total=np.zeros(ds.raw[raw_key].shape[-1],dtype=np.float64);squares=total.copy();count=0;negative_inf=0
  for original in ds.indices:
   valid=ds.support_mask(int(original),raw_key);x=np.asarray(ds.raw[raw_key][int(original)][valid],dtype=np.float64)
   if raw_key=="audio":
    negative_inf+=int(np.isneginf(x).sum());x[np.isneginf(x)]=0.
   if not np.isfinite(x).all():raise ValueError(f"Nonfinite train feature: {ds.dataset}/{raw_key}")
   total+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
  if count==0:raise ValueError("No supported training frames")
  mean=total/count;std=np.sqrt(np.maximum(squares/count-mean**2,1e-12));std=np.maximum(std,1e-6)
  result[name]=dict(mean=mean.tolist(),std=std.tolist(),n_train_frames=count,negative_inf_replaced=negative_inf)
 result.update(fit_split="train_non_neutral_only",view=ds.view,support_convention=ds.support,clip_normalized_features=10.)
 return result
