from pathlib import Path
import gc,math,shutil
import numpy as np
from common import *
import mmap_dataset_pickle as mm
def prepare_split(raw):
 lab=np.asarray(raw['classification_labels']).reshape(-1)
 if not np.isin(lab,[0,1,2]).all():raise ValueError('Expected legacy negative/neutral/positive labels')
 ix=np.flatnonzero(lab!=1);ids=np.asarray(raw['id']).astype(str)[ix];y=(lab[ix]==2).astype(np.int8)
 reg=np.asarray(raw['regression_labels']).reshape(-1)[ix]
 assert np.array_equal(reg>0,y==1)and np.all(reg!=0)and len(np.unique(ids))==len(ids)
 result=dict(ids=ids,y=y,text=np.asarray(raw['text_bert'])[ix].astype(np.int64),source=source_ids(ids),original_index=ix)
 for key in ['audio','vision']:
  x=raw[key];result[key]=np.stack([x[i].mean(0)for i in ix]).astype(np.float32)
  result[key+'_nonzero_rows']=np.asarray([np.any(x[i]!=0,axis=-1).sum()for i in ix],dtype=np.int32)
  if not np.isfinite(result[key]).all():raise ValueError('Nonfinite native feature: '+key)
 for k in ['T','A','V']:
  key='classification_labels_'+k
  result['label_'+k]=np.asarray(raw[key]).reshape(-1)[ix].astype(np.int8)if key in raw else np.full(len(ix),-2,np.int8)
 assert result['text'].ndim==3 and np.isin(result['text'][:,1],[0,1]).all()
 return result
def subset(d,ix):return {k:v[ix]for k,v in d.items()}
def prepare(root,out):
 out=Path(out);p=plan();reference=read(HERE/'provenance/data_reference.json')['inputs'];records={}
 for dataset in p['datasets']:
  path=Path(root)/'MSA Datasets'/REL[dataset];want=next(x['sha256']for x in reference if x['relative_path'].endswith(REL[dataset]))
  digest=sha(path)
  if digest!=want:raise RuntimeError('Dataset identity changed: '+str(path))
  signature=dict(data_sha256=digest,protocol_sha256=sha(HERE/'protocol.json'),bundle_sha256=verify_code())
  target=out/'data'/dataset;meta=target/'prepared.json'
  if meta.exists():
   old=read(meta)
   if old['signature']!=signature:raise RuntimeError('Prepared cache has another protocol; use a new output directory.')
   for name,h in old['files'].items():assert sha(target/name)==h,name
   records[dataset]=old;continue
  with path.open('rb')as f:raw=mm.load(f)
  train=prepare_split(raw['train']);groups=np.unique(train['source'])
  rng=np.random.default_rng(p['split_seed']);groups=rng.permutation(groups);count=max(1,int(math.ceil(len(groups)*p['inner_dev_video_fraction'])))
  audit_count=max(1,int(math.ceil(len(groups)*p['audit_video_fraction'])))
  if count+audit_count>=len(groups):raise ValueError('At least three disjoint source-video groups required')
  devset=set(groups[:count]);auditset=set(groups[count:count+audit_count]);seen=set(groups[count+audit_count:])|devset
  di=np.flatnonzero(np.isin(train['source'],list(devset)));ai=np.flatnonzero(np.isin(train['source'],list(auditset)));fi=np.flatnonzero(~np.isin(train['source'],list(devset|auditset)))
  fit,dev,audit=subset(train,fi),subset(train,di),subset(train,ai)
  assert len(np.unique(audit['y']))==2 and not set(audit['source'])&seen
  if len(np.unique(fit['y']))!=2 or len(np.unique(dev['y']))!=2:raise RuntimeError('Fixed group split lacks a class; do not tune the split silently.')
  assert not set(fit['source'])&set(dev['source'])
  target.mkdir(parents=True,exist_ok=True)
  for name,d in [('fit',fit),('dev',dev),('audit',audit),('valid',prepare_split(raw['valid'])),('test',prepare_split(raw['test']))]:
   d['source_unseen']=~np.isin(d['source'],list(seen));npz(target/f'{name}.npz',**d)
  files={x.name:sha(x)for x in target.glob('*.npz')}
  coverage={}
  for role in ['fit','dev','audit','valid','test']:
   d=arrays(target/f'{role}.npz')
   coverage[role]=dict(n=len(d['y']),source_videos=len(np.unique(d['source'])),source_overlap_with_fit=len(set(d['source'])&set(fit['source'])),source_overlap_with_fit_or_dev=len(set(d['source'])&seen),has_independent_unimodal_labels=bool(np.all(d['label_T']>=0)))
  edges=np.unique(np.quantile(fit['audio_nonzero_rows'],[.2,.4,.6,.8])).tolist()
  record=dict(signature=signature,coverage=coverage,length_bin_edges=edges,split_seed=p['split_seed'],dev_sources=sorted(devset),audit_sources=sorted(auditset),files=files,
   historical_benchmark_data=True,test_used_for_new_training_or_selection=False,test_arrays_prepared_for_integrity_only=True)
  dump(meta,record);records[dataset]=record;del raw,train,fit,dev,audit;gc.collect()
  print('KBS_DATA_PREPARED',dataset,coverage,flush=True)
 dump(out/'data_inventory.json',records);return records
def verify_berts(root,out):
 from transformers import BertConfig
 from bert_weight_check import compare_weights
 configs=read(HERE/'provenance/bert_architecture_reference.json')
 refs=read(HERE/'provenance/input_reference.json');refs=refs['inputs']if isinstance(refs,dict)else refs
 result={}
 for lang in ['bert_en','bert_cn']:
  directory=Path(root)/'MSA Datasets/pretrained_berts'/lang
  config=BertConfig.from_pretrained(str(directory),local_files_only=True).to_dict()
  diff={k:[v,config.get(k)]for k,v in configs[lang].items()if config.get(k)!=v}
  if diff:raise RuntimeError('BERT architecture differs: '+str(diff))
  r=next(x for x in refs if x.get('kind')=='bert_weights'and '/'+lang+'/'in x['relative_path'])
  weights=directory/'model.safetensors'
  if not weights.exists():weights=directory/'pytorch_model.bin'
  check=compare_weights(weights,r['tensor_reference'])
  if not check['passed']:raise RuntimeError('BERT tensors differ: '+str(check))
  result[lang]=dict(path=str(directory),weights_sha256=sha(weights),config_sha256=sha(directory/'config.json'),semantic_architecture_equal=True,tensor_check=check)
  print('KBS_BERT_VERIFIED',lang,check['matched_tensors'],flush=True)
 dump(Path(out)/'bert_identity.json',result);return result
def subsets(d):
 n=len(d['y']);all_=np.ones(n,bool);t=d['label_T'];a=d['label_A'];v=d['label_V'];y=d['y']
 out={'all':all_}
 if 'source_unseen'in d:out['source_unseen']=d['source_unseen'].astype(bool)
 if np.all(t>=0):
  neutral=t==1;aligned=(~neutral)&((t==2)==(y==1));opposed=(~neutral)&(~aligned)
  support=((a!=1)&(a>=0)&((a==2)==(y==1)))|((v!=1)&(v>=0)&((v==2)==(y==1)))
  out.update(text_aligned=aligned,text_opposed=opposed,text_neutral=neutral,nontext_support=(opposed|neutral)&support)
 return out