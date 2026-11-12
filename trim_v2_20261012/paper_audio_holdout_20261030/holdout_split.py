"""Label-independent source-video split; no old fitted predictions are used."""
import hashlib,json
import numpy as np
ROLES=('expert_train','expert_dev','router_train','router_dev')
SPLIT_SEED=20261030

def identity(ids,y):
 return hashlib.sha256(json.dumps([[str(s),int(v)] for s,v in zip(ids,y)],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def make_plan(ids,y,seed=SPLIT_SEED):
 ids=np.asarray(ids);y=np.asarray(y);videos=np.array([str(s).split('$_$')[0] for s in ids])
 names=np.unique(videos)
 if len(names)<20 or len(set(map(str,ids)))!=len(ids):raise ValueError('Need >=20 distinct source videos and unique clip IDs')
 names=np.random.default_rng(seed).permutation(names);n=len(names)
 cuts=[0,round(.675*n),round(.75*n),round(.90*n),n]
 roles={}
 for i,role in enumerate(ROLES):
  selected=np.sort(names[cuts[i]:cuts[i+1]]);ix=np.flatnonzero(np.isin(videos,selected))
  counts=np.bincount(y[ix].astype(np.int64),minlength=2)
  if len(selected)<1 or counts.min()<1:raise ValueError('Both classes required; do not resample based on validation')
  roles[role]=dict(indices=ix.tolist(),videos=selected.tolist(),n=len(ix),class_counts=counts.tolist(),ids_sha256=identity(ids[ix],y[ix]))
 result=dict(seed=seed,assignment='sorted_unique_video_then_fixed_rng_permutation_no_labels',video_fractions=[.675,.075,.15,.10],
  parent_train_n=len(ids),parent_train_videos=n,parent_train_ids_sha256=identity(ids,y),roles=roles,
  external_validation_for_selection=False,test_evaluated=False)
 validate_plan(result,ids,y);return result

def validate_plan(plan,ids,y):
 ids=np.asarray(ids);y=np.asarray(y);videos=np.array([str(s).split('$_$')[0] for s in ids]);seen=set();seen_videos=set()
 if identity(ids,y)!=plan['parent_train_ids_sha256']:raise RuntimeError('Parent train sample identity changed')
 if set(plan['roles'])!=set(ROLES):raise RuntimeError('Unexpected split roles')
 for role in ROLES:
  r=plan['roles'][role];ix=np.asarray(r['indices'],dtype=np.int64);vs=set(r['videos'])
  if len(ix)!=len(set(ix.tolist())) or np.any(ix<0) or np.any(ix>=len(ids)):raise RuntimeError('Invalid role index')
  if seen.intersection(ix.tolist()) or seen_videos.intersection(vs):raise RuntimeError('Role sample/video overlap')
  if set(videos[ix])!=vs or not np.array_equal(ix,np.flatnonzero(np.isin(videos,list(vs)))):raise RuntimeError('Source video split is incomplete')
  if len(ix)!=r['n'] or np.bincount(y[ix].astype(np.int64),minlength=2).tolist()!=r['class_counts'] or identity(ids[ix],y[ix])!=r['ids_sha256']:
   raise RuntimeError('Role samples or labels differ')
  seen.update(ix.tolist());seen_videos.update(vs)
 if seen!=set(range(len(ids))):raise RuntimeError('Roles do not partition the full training subset')
 return True