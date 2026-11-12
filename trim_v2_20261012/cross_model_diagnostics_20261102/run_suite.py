"""Sequential fixed-checkpoint replay, with validated per-job resume."""
from pathlib import Path
import argparse,subprocess,sys,time
from diag_common import HERE,SEEDS,METHODS,DATASETS,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=HERE.parents[1]);p.add_argument('--out',type=Path);p.add_argument('--datasets',nargs='+',default=list(DATASETS),choices=['MOSI','SIMS','MOSEI']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 out=a.out or HERE/'runs'/('smoke' if a.smoke else 'diagnostic');out=out.resolve();out.mkdir(parents=True,exist_ok=True)
 datasets=['MOSI'] if a.smoke else a.datasets;seeds=[42] if a.smoke else SEEDS
 jobs=[(d,m,s)for d in datasets for m in METHODS for s in seeds];started=time.time()
 for i,(d,m,s) in enumerate(jobs):
  cmd=[sys.executable,'-u',str(HERE/'worker.py'),'--root',str(a.root),'--out',str(out),'--dataset',d,'--method',m,'--seed',str(s)]
  if a.smoke:cmd.append('--smoke')
  print('DIAG_SUITE_JOB',i+1,'/',len(jobs),d,m,s,flush=True);subprocess.run(cmd,check=True,cwd=HERE)
  elapsed=time.time()-started;dump(out/'suite_progress.json',dict(completed=i+1,total=len(jobs),seconds=elapsed,last_job=[d,m,s],smoke=a.smoke))
  print('DIAG_SUITE_PROGRESS',i+1,'/',len(jobs),'seconds',round(elapsed,1),flush=True)
 if a.smoke:print('DIAG_SMOKE_COMPLETE prefix_only; not a scientific result',flush=True)
 else:
  subprocess.run([sys.executable,'-u',str(HERE/'summarize.py'),'--out',str(out),'--datasets',*datasets],check=True,cwd=HERE)
  print('DIAG_SUITE_COMPLETE',str(out/'cross_model_review_20261102.zip'),flush=True)
if __name__=='__main__':main()