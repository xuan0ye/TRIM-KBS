"""Report model-specific intervention effects, preserving the seed hierarchy."""
from pathlib import Path
import argparse,csv,json,zipfile
import numpy as np
from diag_common import HERE,SEEDS,METHODS,DATASETS,sha_file,dump,csvout

PRIMARY=['audio_cross_video','vision_cross_video','paired_av_cross_video']

def macro_counts(c):
 tn,fp,fn,tp=np.moveaxis(c,-1,0)
 return (2*tn/np.maximum(1,2*tn+fp+fn)+2*tp/np.maximum(1,2*tp+fp+fn))/2

def bootstrap_effect(y,base,intervened,ids,resamples=2000,seed=20261102):
 # Every resample selects whole source videos. The same draw is shared by conditions and all fixed checkpoints.
 _,video=np.unique(np.asarray([str(i).split('$_$')[0] for i in ids]),return_inverse=True);nv=int(video.max()+1)
 def counts(pred):
  result=np.zeros((pred.shape[0],pred.shape[1],nv,4),dtype=np.float64)
  for s in range(len(pred)):
   for r in range(pred.shape[1]):np.add.at(result[s,r],(video,2*y+pred[s,r]),1)
  return result
 base_c=counts(base[:,None,:]);new_c=counts(intervened)
 rng=np.random.default_rng(seed);weights=rng.multinomial(nv,np.full(nv,1/nv),size=resamples)
 bc=np.einsum('bv,srvc->bsrc',weights,base_c,optimize=True);ic=np.einsum('bv,srvc->bsrc',weights,new_c,optimize=True)
 delta=100*(macro_counts(ic).mean(axis=(1,2))-macro_counts(bc).mean(axis=(1,2)))
 lo,hi=np.quantile(delta,[.025,.975]);return float(lo),float(hi),nv

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=HERE/'runs/diagnostic');p.add_argument('--datasets',nargs='+',default=list(DATASETS),choices=['MOSI','SIMS','MOSEI']);a=p.parse_args();out=a.out.resolve()
 summaries=[];all_rows=[];records=[];bundle_digest=sha_file(HERE/'bundle_manifest.json')
 for dataset in a.datasets:
  for method in METHODS:
   grouped={};arrays=[]
   for seed in SEEDS:
    folder=out/dataset/method/f'seed_{seed}';metadata=json.loads((folder/'complete.json').read_text('utf8'))
    if metadata['smoke']:raise RuntimeError('Smoke predictions cannot enter scientific tables')
    if metadata['signature']['bundle_sha256']!=bundle_digest:raise RuntimeError('Mixed or different source bundle in results')
    if (metadata['signature']['dataset'],metadata['signature']['method'],metadata['signature']['seed'])!=(dataset,method,seed):raise RuntimeError('Job metadata does not match its folder')
    for filename,digest in metadata['output_sha256'].items():
     if sha_file(folder/filename)!=digest:raise RuntimeError('Result changed: '+str(folder/filename))
    records.append(dict(dataset=dataset,method=method,seed=seed,checkpoint_sha256=metadata['signature']['checkpoint_sha256'],historical_hash_available=metadata['historical_checkpoint_hash_available']))
    with np.load(folder/'predictions.npz',allow_pickle=False) as z:arrays.append({k:z[k] for k in ['ids','labels','condition_names','predictions','native_scores']})
    with (folder/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    all_rows.extend(rows)
    for row in rows:grouped.setdefault(row['family'],{}).setdefault(seed,[]).append(row)
   first=arrays[0]
   for z in arrays[1:]:
    for key in ['ids','labels','condition_names']:
     if not np.array_equal(z[key],first[key]):raise RuntimeError('Paired summary requires identical ordered samples and conditions across training seeds')
   for family,by_seed in grouped.items():
    result=dict(dataset=dataset,method=method,family=family,training_seeds=len(by_seed),draws_per_seed=len(by_seed[SEEDS[0]]),n=len(first['labels']),
     score_type=metadata['score_type'],temporal_note=metadata['temporal_note'] if family.endswith('_order') else '')
    for key in ['macro_f1','f1_delta_pp','accuracy','changed_fraction','corrected','broken','mean_abs_native_score_delta','max_abs_native_score_delta']:
     values=np.asarray([np.mean([float(r[key]) for r in by_seed[s]]) for s in SEEDS])
     result[key+'_mean']=float(values.mean());result[key+'_sd_across_training_seeds']=float(values.std(ddof=1))
    if family in PRIMARY+['text_only']:
     names=first['condition_names'];indices=[i for i,name in enumerate(names) if name==family or str(name).startswith(family+'_')]
     baseline=np.stack([z['predictions'][0] for z in arrays]);new=np.stack([z['predictions'][indices] for z in arrays])
     lo,hi,nv=bootstrap_effect(first['labels'],baseline,new,first['ids']);result.update(video_bootstrap_f1_delta_lo_pp=lo,video_bootstrap_f1_delta_hi_pp=hi,source_videos=nv)
    summaries.append(result)
 csvout(out/'condition_by_seed_and_draw.csv',all_rows);csvout(out/'summary.csv',summaries)
 lines=['# Cross-model input-intervention diagnostics','',
  'All five archived checkpoints per architecture were replayed without fitting or threshold selection. Results reuse previously examined benchmark tests. Random intervention draws are averaged within each training seed; standard deviations are across the five training seeds.','',
  '## Primary effects','',
  'Macro-F1 differences are intervention minus full input, in percentage points. Decision changes are percentages. Bootstrap intervals resample source videos jointly across fixed checkpoints and intervention draws (2,000 draws); they are conditional on these archived checkpoints.','',
  '|Dataset|System|Intervention|Macro-F1 difference, mean ± SD|Video bootstrap 95% interval|Decision changes|','|---|---|---|---:|---:|---:|']
 for r in summaries:
  if r['family'] not in PRIMARY+['text_only']:continue
  lines.append(f"|{r['dataset']}|{r['method']}|{r['family']}|{r['f1_delta_pp_mean']:+.4f} ± {r['f1_delta_pp_sd_across_training_seeds']:.4f}|[{r['video_bootstrap_f1_delta_lo_pp']:+.4f}, {r['video_bootstrap_f1_delta_hi_pp']:+.4f}]|{100*r['changed_fraction_mean']:.4f}%|")
 lines+=['','## Interpretation constraints','',
  'TRIM receives pooled acoustic and visual features. Permuting native time steps before its mean pooling leaves those inputs unchanged; its temporal conditions are marked structural_identity_after_pooled_inputs. Cross-video replacement changes the pooled features and is the informative counterpart.','',
  'For sequence models, temporal permutation only rearranges nonzero-row proxy positions and preserves zero rows. Nonzero positions are not asserted to be verified speech boundaries or true padding masks. Audio and visual sequence orders are shuffled separately.','',
  'TRIM and MulT score shifts use positive-class probabilities. LNLN shifts use raw regression scores with a fixed >0 classification decision. Their score magnitudes cannot be ranked on a shared probability scale. Corrected and broken decisions must be read together with aggregate F1.','',
  'A text-only intervention on a multimodal checkpoint is distinct from independently training a text-only model. MulT checkpoints have no archived local SHA256 record: this bundle records their current digest and requires exact four-condition historical decision replay; it does not claim historical tensor-identity verification for those weights.','',
  'No causal claim about human emotion, no fresh confirmatory test, and no model improvement is established by these diagnostic interventions.']
 (out/'REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 dump(out/'review_audit.json',dict(jobs=len(records),expected_jobs=len(a.datasets)*len(METHODS)*len(SEEDS),records=records,
  datasets=a.datasets,methods=METHODS,training_seeds=SEEDS,permutation_repetitions_not_training_seeds=True,previously_examined_test=True,
  bootstrap_resamples=2000,bootstrap_unit='source video',new_training=False,source_bundle_sha256=sha_file(HERE/'bundle_manifest.json')))
 files=[out/'summary.csv',out/'condition_by_seed_and_draw.csv',out/'REVIEW.md',out/'review_audit.json']
 for r in records:
  folder=out/r['dataset']/r['method']/f"seed_{r['seed']}";files.extend(folder/name for name in ['complete.json','metrics.csv','predictions.npz'])
 members={p.relative_to(out).as_posix():sha_file(p) for p in files}
 members['analysis_plan.json']=sha_file(HERE/'analysis_plan.json');members['source_bundle_manifest.json']=bundle_digest
 archive=out/'cross_model_review_20261102.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for path in files:z.write(path,path.relative_to(out).as_posix())
  z.write(HERE/'analysis_plan.json','analysis_plan.json');z.write(HERE/'bundle_manifest.json','source_bundle_manifest.json')
  z.writestr('review_member_manifest.json',json.dumps(members,indent=2))
 print('DIAG_REVIEW_COMPLETE',archive,flush=True)
if __name__=='__main__':main()