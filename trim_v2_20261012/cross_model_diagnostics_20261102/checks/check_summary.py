"""Exercise the complete summary and return ZIP using explicitly synthetic fixtures."""
from pathlib import Path
import csv,hashlib,json,sys,tempfile,zipfile
import numpy as np
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
from diag_common import cases,compare,csvout,dump,sha_file,SEEDS,METHODS
import summarize

def main():
 check=HERE/'checks'
 with tempfile.TemporaryDirectory(prefix='synthetic_summary_',dir=str(check)) as tmp:
  out=Path(tmp).resolve();assert out.is_relative_to(check.resolve())
  ids=np.asarray(['a$_$0','a$_$1','b$_$0','b$_$1','c$_$0','c$_$1']);y=np.array([0,1,0,1,0,1],np.int8)
  definitions,donors=cases(ids);base=np.array([0,1,1,0,0,1],np.int8)
  for dataset in ['MOSI','SIMS']:
   for method in METHODS:
    for seed in SEEDS:
     folder=out/dataset/method/f'seed_{seed}';folder.mkdir(parents=True)
     pred=np.stack([base.copy() for _ in definitions]);pred[1:,2]=0;scores=pred.astype(np.float32)*.8+.1
     rows=[]
     for j,c in enumerate(definitions):rows.append(dict(dataset=dataset,method=method,seed=seed,condition=c['name'],family=c['kind'],rep=c['rep'],score_type='synthetic_fixture',temporal_note='',**compare(y,base,pred[j],scores[0],scores[j])))
     np.savez_compressed(folder/'predictions.npz',ids=ids,labels=y,condition_names=np.asarray([c['name'] for c in definitions]),predictions=pred,native_scores=scores)
     csvout(folder/'metrics.csv',rows)
     dump(folder/'complete.json',dict(smoke=False,signature=dict(bundle_sha256=sha_file(HERE/'bundle_manifest.json'),checkpoint_sha256='synthetic-fixture',dataset=dataset,method=method,seed=seed),historical_checkpoint_hash_available=False,score_type='synthetic_fixture',temporal_note='synthetic_fixture',output_sha256={n:sha_file(folder/n)for n in ['predictions.npz','metrics.csv']}))
  sys.argv=['summarize.py','--out',str(out)];summarize.main()
  with (out/'summary.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
  assert len(rows)==78
  primary=[r for r in rows if r['family']=='audio_cross_video'];assert len(primary)==6
  for r in primary:
   assert int(r['training_seeds'])==5 and int(r['draws_per_seed'])==20
   assert abs(float(r['f1_delta_pp_mean'])-100*(compare(y,base,pred[1],scores[0],scores[1])['macro_f1']-compare(y,base,base,scores[0],scores[0])['macro_f1']))<1e-10
  with zipfile.ZipFile(out/'cross_model_review_20261102.zip') as z:
   hashes=json.loads(z.read('review_member_manifest.json'))
   for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
  # A modified result must be rejected before it can be summarized.
  target=out/'MOSI/TRIM/seed_13/metrics.csv';target.write_text('modified\n',encoding='utf8')
  try:summarize.main();raise AssertionError('Changed result was accepted')
  except RuntimeError as e:assert 'Result changed' in str(e)
 dump(check/'summary_check.json',dict(status='passed',synthetic_only=True,fixture_jobs=30,summary_rows=78,training_seeds=5,draws_per_seed=20,return_zip_hashes_verified=True,modified_results_rejected=True,fixtures_removed=True))
 print('SUMMARY_SYNTHETIC_CHECK_PASSED')
if __name__=='__main__':main()