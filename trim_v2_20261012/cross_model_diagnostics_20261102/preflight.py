"""Check availability before starting a long remote replay; no downloads or training."""
from pathlib import Path
import argparse,importlib.util,json,sys
from diag_common import HERE,METHODS,SEEDS,DATASETS,verify_bundle,checkpoint,dump
from input_identity import data_path,verify_bert

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=HERE.parents[1]);p.add_argument('--datasets',nargs='+',default=list(DATASETS),choices=['MOSI','SIMS','MOSEI']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 root=a.root.resolve();errors=[];records=[]
 try:bundle=verify_bundle()
 except Exception as e:errors.append(str(e));bundle=None
 for module in ['torch','transformers','numpy','yaml','einops']:
  if importlib.util.find_spec(module) is None:errors.append('Missing Python module: '+module)
 try:
  import torch
  if not torch.cuda.is_available():errors.append('CUDA unavailable in '+sys.executable)
  else:print('DIAG_GPU',torch.cuda.get_device_name(),flush=True)
 except Exception as e:errors.append('Torch import failed: '+str(e))
 datasets=['MOSI'] if a.smoke else a.datasets;seeds=[42] if a.smoke else SEEDS
 for dataset in datasets:
  for method in METHODS:
   try:
    data=data_path(root,dataset,method)
    if not data.is_file():raise FileNotFoundError(data)
    bert=verify_bert(root,dataset,method)
    for seed in seeds:
     path,digest,pinned=checkpoint(root,dataset,method,seed)
     records.append(dict(dataset=dataset,method=method,seed=seed,checkpoint=str(path),sha256=digest,historical_hash_available=pinned))
   except Exception as e:errors.append(f'{dataset}/{method}: {e}')
 report=dict(root=str(root),python=sys.executable,bundle_sha256=bundle,checkpoint_records=records,errors=errors,
  data_hash_check='streamed and enforced again inside every actual job',smoke=a.smoke)
 dump(HERE/'logs'/('preflight_smoke.json' if a.smoke else 'preflight.json'),report)
 for e in errors:print('DIAG_PREFLIGHT_ERROR',e,flush=True)
 if errors:raise SystemExit(2)
 print('DIAG_PREFLIGHT_OK',len(records),'checkpoints; no optimizer or new training',flush=True)
if __name__=='__main__':main()