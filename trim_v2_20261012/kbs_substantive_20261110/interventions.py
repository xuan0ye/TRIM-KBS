"""Label-independent perturbations and logit-space EMAP for T versus paired A/V."""
import hashlib
import numpy as np
def cross_source(source,seed):
 source=np.asarray(source);names,counts=np.unique(source,return_counts=True);n=len(source)
 if n<2 or counts.max()>n//2:return np.full(n,-1,np.int64)
 rng=np.random.default_rng(seed)
 order=np.concatenate([rng.permutation(np.flatnonzero(source==v))for v in rng.permutation(names)])
 donor=np.empty(n,np.int64);donor[order]=np.roll(order,int(rng.integers(counts.max(),n-counts.max()+1)))
 assert len(np.unique(donor))==n and np.all(source[donor]!=source)
 return donor
def within_source(source,seed):
 rng=np.random.default_rng(seed);out=np.full(len(source),-1,np.int64)
 for name in np.unique(source):
  ix=np.flatnonzero(source==name)
  if len(ix)<2:continue
  order=rng.permutation(ix);out[order]=np.roll(order,int(rng.integers(1,len(ix))))
 ix=np.flatnonzero(out>=0)
 assert np.all(source[out[ix]]==source[ix])and np.all(out[ix]!=ix)
 return out
def length_matched(source,length,edges,seed):
 labels=np.digitize(length,edges);bins=[np.flatnonzero(labels==k)for k in np.unique(labels)]
 def possible(ix):
  return len(ix)>=2 and np.unique(np.asarray(source)[ix],return_counts=True)[1].max()<=len(ix)//2
 while len(bins)>1:
  bad=next((k for k,b in enumerate(bins)if not possible(b)),None)
  if bad is None:break
  neighbour=bad-1 if bad else 1;lo,hi=sorted([bad,neighbour]);bins[lo]=np.r_[bins[lo],bins[hi]];bins.pop(hi)
 out=np.full(len(source),-1,np.int64);stratum=np.full(len(source),-1,np.int16)
 for k,ix in enumerate(bins):
  donor=cross_source(np.asarray(source)[ix],seed+k*7919)
  if np.all(donor>=0):out[ix]=ix[donor];stratum[ix]=k
 return out,stratum
def build_maps(data,edges,repeats=20,seed=20261110):
 source=data['source'];length=data['audio_nonzero_rows'];items={}
 for rep in range(repeats):
  items[f'cross_{rep:02}']=cross_source(source,seed+rep*101)
  items[f'within_{rep:02}']=within_source(source,seed+rep*101+1)
  donor,bins=length_matched(source,length,edges,seed+rep*101+2)
  items[f'length_{rep:02}']=donor
 return items
def exact_emap(n,predict,pair_batch=256):
 row=np.zeros(n,np.float64);col=np.zeros(n,np.float64);total=0.;squares=0.
 for start in range(0,n*n,pair_batch):
  flat=np.arange(start,min(start+pair_batch,n*n));i,j=flat//n,flat%n
  value=np.asarray(predict(i,j),np.float64);assert value.shape==i.shape and np.isfinite(value).all()
  np.add.at(row,i,value);np.add.at(col,j,value);total+=value.sum();squares+=np.square(value).sum()
 row/=n;col/=n;grand=total/(n*n)
 projected=row+col-grand
 residual=max(0.,squares/(n*n)-np.square(row).mean()-np.square(col).mean()+grand*grand)
 return projected,dict(kind='exact_empirical_product',pairs=n*n,grid_interaction_rms=float(np.sqrt(residual)))
def anchor_emap(n,anchors,predict,pair_batch=256):
 anchors=np.asarray(anchors);m=len(anchors);r=np.empty((n,m),np.float64);c=np.empty((n,m),np.float64)
 for start in range(0,n*m,pair_batch):
  flat=np.arange(start,min(start+pair_batch,n*m));i,j=flat//m,flat%m
  r[i,j]=predict(i,anchors[j]);c[i,j]=predict(anchors[j],i)
 grid=np.empty((m,m),np.float64)
 for start in range(0,m*m,pair_batch):
  flat=np.arange(start,min(start+pair_batch,m*m));i,j=flat//m,flat%m
  grid[i,j]=predict(anchors[i],anchors[j])
 full=r.mean(1)+c.mean(1)-grid.mean();h=max(1,m//2)
 half=r[:,:h].mean(1)+c[:,:h].mean(1)-grid[:h,:h].mean()
 return full,half,dict(kind='Monte_Carlo_empirical_product',anchors=m,pairs=2*n*m+m*m,half_to_full_margin_rmse=float(np.sqrt(np.square(half-full).mean())))
def anchors_for(ids,n=64,banks=2,seed=20261110):
 rng=np.random.default_rng(seed);return [rng.choice(len(ids),min(n,len(ids)),replace=False)for _ in range(banks)]
def f1_counts(c):
 tn,fp,fn,tp=np.moveaxis(np.asarray(c,dtype=float),-1,0)
 return .5*(2*tn/np.maximum(1,2*tn+fp+fn)+2*tp/np.maximum(1,2*tp+fp+fn))
def clustered_counts(y,margin,source,eligible):
 names=np.unique(source);out=np.zeros((len(names),4),np.float64)
 q=(np.asarray(margin)>0).astype(int)
 for k,v in enumerate(names):
  ix=(source==v)&eligible;out[k]=np.bincount(2*y[ix].astype(int)+q[ix],minlength=4)
 return out
def paired_bootstrap(base,changed,y,source,eligible,repetitions=2000,seed=20261110):
 # Arrays: seeds x draws x examples. Draws remain nested within each training seed.
 base=np.asarray(base);changed=np.asarray(changed)
 if base.ndim==2:base=base[:,None,:]
 if changed.ndim==2:changed=changed[:,None,:]
 s,r,n=changed.shape;assert base.shape[0]==s and base.shape[-1]==n
 names=np.unique(source);v=len(names)
 cb=np.stack([clustered_counts(y,b[0],source,eligible)for b in base])
 cc=np.stack([[clustered_counts(y,x,source,eligible)for x in z]for z in changed])
 point=(f1_counts(cc.sum(2)).mean(1)-f1_counts(cb.sum(1))).mean()*100
 rng=np.random.default_rng(seed);conditional=[];crossed=[]
 for start in range(0,repetitions,64):
  b=min(64,repetitions-start);weights=rng.multinomial(v,np.full(v,1/v),size=b)
  mb=f1_counts(np.einsum('svk,bv->bsk',cb,weights))
  mc=f1_counts(np.einsum('srvk,bv->bsrk',cc,weights)).mean(-1)
  delta=mc-mb;conditional.extend(delta.mean(1)*100)
  ix=rng.integers(0,s,size=(b,s));crossed.extend(np.take_along_axis(delta,ix,axis=1).mean(1)*100)
 return dict(delta_pp=float(point),video_conditional_ci95=list(map(float,np.quantile(conditional,[.025,.975]))),
  video_and_seed_ci95=list(map(float,np.quantile(crossed,[.025,.975]))),bootstrap_repetitions=repetitions,seeds=s,source_groups=v,
  interpretation='intervals conditional on fixed protocol; not evidence of new untouched external data')