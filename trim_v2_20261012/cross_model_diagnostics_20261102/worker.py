"""One frozen architecture/seed, all predeclared input interventions."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from pathlib import Path
import argparse,json,time,platform,sys
import numpy as np
import torch
import transformers,yaml,einops
from diag_common import HERE,sha_file,dump,csvout,cases,compare,verify_bundle,checkpoint
from input_identity import verify_data,verify_bert

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=HERE.parents[1]);p.add_argument('--out',type=Path,default=HERE/'runs/diagnostic')
 p.add_argument('--dataset',required=True,choices=['MOSI','SIMS','MOSEI']);p.add_argument('--method',required=True,choices=['TRIM','MulT','LNLN']);p.add_argument('--seed',type=int,required=True)
 p.add_argument('--batch-size',type=int,default=64);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true')
 a=p.parse_args();a.root=a.root.resolve();a.out=a.out.resolve()
 if a.batch_size<2:raise ValueError('Batch size must be >=2 (native LNLN squeeze behaviour).')
 torch.set_num_threads(4);torch.manual_seed(20261102);np.random.seed(20261102)
 torch.backends.cudnn.enabled=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; use the configured GPU Python environment.')
 started=time.time();bundle=verify_bundle();ckpt,digest,historical_hash=checkpoint(a.root,a.dataset,a.method,a.seed)
 native_data=verify_data(a.root,a.dataset,a.method);bert=verify_bert(a.root,a.dataset,a.method)
 signature=dict(bundle_sha256=bundle,checkpoint_sha256=digest,data=native_data,dataset=a.dataset,method=a.method,seed=a.seed,
  batch_size=a.batch_size,smoke=a.smoke,device=a.device,torch=torch.__version__,transformers=transformers.__version__,
  numpy=np.__version__,yaml=yaml.__version__,einops=einops.__version__,cuda=torch.version.cuda,
  gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None)
 folder=a.out/a.dataset/a.method/f'seed_{a.seed}';folder.mkdir(parents=True,exist_ok=True);complete=folder/'complete.json'
 if complete.is_file():
  previous=json.loads(complete.read_text('utf8'))
  if previous['signature']!=signature:raise RuntimeError('Existing result has different inputs/runtime. Use a new output directory: '+str(folder))
  for name,expected in previous['output_sha256'].items():
   if sha_file(folder/name)!=expected:raise RuntimeError('Resume output hash mismatch: '+str(folder/name))
  print('DIAG_RESUME_VERIFIED',a.dataset,a.method,a.seed,flush=True);return
 from adapters import Adapter
 adapter=Adapter(a.root,a.dataset,a.method,ckpt,torch.device(a.device))
 ref_path=HERE/'reference'/a.dataset/a.method/f'seed_{a.seed}.npz'
 with np.load(ref_path,allow_pickle=False) as f:ref={k:f[k] for k in f.files}
 if not np.array_equal(adapter.ids.astype(str),ref['ids'].astype(str)):raise RuntimeError('Native ordered sample IDs differ from reference')
 if not np.array_equal(adapter.y,ref['labels']):raise RuntimeError('Native labels differ from reference')
 definitions,donors=cases(adapter.ids);n_all=len(adapter.y);n=min(8,n_all) if a.smoke else n_all
 pred=np.empty((len(definitions),n),np.int8);scores=np.empty((len(definitions),n),np.float32)
 checks={};reference_mismatches={k:0 for k in ['full','text_only','text_audio','text_visual']};max_prob_error=0.
 # Native LNLN has a scalar squeeze for batch size one. Merge the final singleton.
 batches=[np.arange(st,min(st+a.batch_size,n)) for st in range(0,n,a.batch_size)]
 if len(batches)>1 and len(batches[-1])==1:
  last=batches.pop();batches[-1]=np.concatenate([batches[-1],last])
 print('DIAG_JOB_START',a.dataset,a.method,a.seed,'n',n,'conditions',len(definitions),'historical_checkpoint_hash',historical_hash,flush=True)
 with torch.inference_mode():
  for batch_no,ix in enumerate(batches):
   adapter.start(ix)
   if batch_no==0:checks=adapter.check_native(definitions,donors)
   for j,case in enumerate(definitions):
    pr,sc,_=adapter.predict(case,donors);pred[j,ix]=pr;scores[j,ix]=sc
    if case['name'] in reference_mismatches:
     errors=int((pr!=ref[case['name']][ix]).sum());reference_mismatches[case['name']]+=errors
     if errors:raise RuntimeError(f'Historical reference mismatch: {case["name"]}, batch {batch_no}, {errors} decisions. No fallback model/checkpoint selection is permitted.')
    if a.method=='TRIM' and case['name']=='full' and 'prob_positive' in ref:
     difference=float(np.abs(sc-ref['prob_positive'][ix]).max());max_prob_error=max(max_prob_error,difference)
     if difference>1e-4:raise RuntimeError('TRIM full-input probability replay differs: '+str(difference))
   elapsed=time.time()-started
   dump(folder/'progress.json',dict(dataset=a.dataset,method=a.method,seed=a.seed,batches_done=batch_no+1,batches_total=len(batches),seconds=elapsed))
   print('DIAG_BATCH',a.dataset,a.method,a.seed,batch_no+1,'/',len(batches),'seconds',round(elapsed,1),flush=True)
 y=adapter.y[:n];rows=[]
 for j,c in enumerate(definitions):
  rows.append(dict(dataset=a.dataset,method=a.method,seed=a.seed,condition=c['name'],family=c['kind'],rep=c['rep'],score_type=adapter.score_type,
   temporal_note=adapter.temporal_note if c['kind'].endswith('_order') else '',**compare(y,pred[0],pred[j],scores[0],scores[j])))
 extra={}
 for name,(da,dv) in donors.items():extra['donor_audio__'+name]=da;extra['donor_vision__'+name]=dv
 np.savez_compressed(folder/'predictions.npz',ids=adapter.ids[:n].astype(str),labels=y,condition_names=np.asarray([c['name'] for c in definitions]),
  predictions=pred,native_scores=scores,donor_population_ids=adapter.ids.astype(str),**extra)
 csvout(folder/'metrics.csv',rows)
 # This is replay on previously examined benchmark tests, not a fresh confirmatory test.
 metadata=dict(signature=signature,reference_sha256=sha_file(ref_path),checkpoint_path=str(ckpt),historical_checkpoint_hash_available=historical_hash,
  functional_reference_replay='all_four_conditions_exact_decisions' if not a.smoke else 'prefix_only_smoke',reference_mismatches=reference_mismatches,
  bert=bert,wrapper_equivalence=checks,full_probability_max_error=max_prob_error if a.method=='TRIM' else None,
  parameters=adapter.parameters,score_type=adapter.score_type,temporal_note=adapter.temporal_note,n=n,total_native_examples=n_all,
  input_shapes=dict(text=list(adapter.t.shape),audio=list(adapter.a.shape),vision=list(adapter.v.shape)),
  temporal_examples_with_any_input_change={c['name']:len(adapter.temporal_changed.get(c['name'],set())) for c in definitions if c['kind'].endswith('_order')},
  conditions=len(definitions),smoke=a.smoke,seconds=time.time()-started,bert_memo_calls=adapter.memo_calls,bert_memo_hits=adapter.memo_hits,
  environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,yaml=yaml.__version__,einops=einops.__version__,cuda=torch.version.cuda,
   gpu=torch.cuda.get_device_name() if a.device.startswith('cuda') else None),
  test_evaluated=True,previously_examined_test=True,new_training=False,selection_performed=False,
  output_sha256={name:sha_file(folder/name) for name in ['predictions.npz','metrics.csv']})
 dump(complete,metadata);print('DIAG_JOB_COMPLETE',a.dataset,a.method,a.seed,'seconds',round(metadata['seconds'],1),'smoke',a.smoke,flush=True)
if __name__=='__main__':main()