"""Synthetic end-to-end check; temporary predictions/checkpoints are discarded."""
from pathlib import Path
import copy,tempfile,json,hashlib
import numpy as np
import torch
import common,train,evaluate,summarize,data,models
from common import dump,npz,sha,HERE
def main():
 synthetic=copy.deepcopy(common.plan());synthetic.update(datasets=['SIMS'],seeds=[11,12],phases={'SIMS':[1]},batch_size=4,accumulation_steps=2)
 synthetic['interventions']['repeats']=2;synthetic['bootstrap']['repeats']=32
 for module in [common,train,evaluate,summarize,data]:module.plan=lambda:synthetic
 original=models.make_model
 def tiny(bert,a_dim,v_dim,variant,**kwargs):
  kwargs.update(tiny=True,d_model=32);return original(bert,a_dim,v_dim,variant,**kwargs)
 train.make_model=tiny;evaluate.make_model=tiny
 rng=np.random.default_rng(3);y=np.array([0,1]*6,np.int8);text=rng.integers(0,64,(12,3,6),dtype=np.int64);text[:,1]=1;text[:,2]=0
 aux=np.array([0,2,2,0,1,1,0,2,2,0,1,1],np.int8)
 with tempfile.TemporaryDirectory(prefix='kbs_synthetic_')as tmp:
  out=Path(tmp);dd=out/'data/SIMS';dd.mkdir(parents=True)
  for role in ['fit','dev','audit','valid','test']:
   source=np.repeat([role+str(k)for k in range(4)],3)
   ids=np.asarray([s+'$_$'+str(k)for k,s in enumerate(source)])
   d=dict(ids=ids,y=y,text=text,audio=rng.normal(size=(12,5)).astype('float32'),vision=rng.normal(size=(12,7)).astype('float32'),source=source,
    audio_nonzero_rows=np.arange(12,dtype=np.int32),vision_nonzero_rows=np.arange(12,dtype=np.int32),label_T=aux,label_A=(y*2).astype(np.int8),label_V=(y*2).astype(np.int8),source_unseen=np.full(12,role in ['audit','valid','test'],bool))
   npz(dd/f'{role}.npz',**d)
  files={p.name:sha(p)for p in dd.glob('*.npz')};dump(dd/'prepared.json',dict(signature={'synthetic_only':True},files=files,length_bin_edges=[2,4,7]))
  dump(out/'bert_identity.json',{'synthetic_tiny_bert_only':True});dump(out/'data_inventory.json',{'synthetic_only':True})
  for seed in synthetic['seeds']:
   for variant in synthetic['variants']:train.train_one(Path(tmp),out,'SIMS',variant,seed,'cpu')
  train.seal_training(out)
  for seed in synthetic['seeds']:
   for variant in synthetic['variants']:
    for role in synthetic['evaluation_roles']:evaluate.evaluate_one(Path(tmp),out,'SIMS',variant,seed,role,'cpu')
  archive=summarize.summarize(out);assert archive.is_file()
  # Resume/cache paths must not retrain or silently produce another checkpoint.
  checkpoint=out/'training/SIMS/trim/seed_11/best.pt';before=sha(checkpoint)
  train.train_one(Path(tmp),out,'SIMS','trim',11,'cpu');assert sha(checkpoint)==before
  evaluate.evaluate_one(Path(tmp),out,'SIMS','trim',11,'test','cpu')
  audit=common.read(out/'review/review_audit.json');assert audit['parameter_and_stochastic_pairing_verified']
  assert audit['new_training_jobs']==10
 report=dict(status='passed',synthetic_training_jobs=10,synthetic_evaluation_jobs=30,formal_benchmark_training_jobs=0,formal_benchmark_evaluation_jobs=0,
  temporary_checkpoint_and_prediction_files_discarded=True,cache_and_seal_paths_checked=True,paired_randomness_verified=True)
 dump(HERE/'checks/integration_engineering_check.json',report);print('KBS_INTEGRATION_ENGINEERING_PASSED; synthetic temporary data discarded',flush=True)
if __name__=='__main__':main()