"""Summaries keep masks/donor draws inside training seeds and report paired intervals."""
from pathlib import Path
import csv,json,zipfile
import numpy as np
from common import *
from data import subsets
from interventions import paired_bootstrap
def summarize(out):
 out=Path(out);p=plan();performance=[];effects=[];comparisons=[];budget=[];complete_index={}
 # Verify seed pairing and initialization before reading any model ranking.
 for dataset in p['datasets']:
  for seed in p['seeds']:
   identities=[];mask_sequences=[];orders=[]
   for variant in p['variants']:
    folder=out/'training'/dataset/variant/f'seed_{seed}';m=read(folder/'model_identity.json');budget.append(dict(dataset=dataset,seed=seed,**m))
    if variant!='bert_mlp':identities.append(m['common_branch_initialization_sha256'])
    with (folder/'history.csv').open(encoding='utf-8-sig',newline='')as f:h=list(csv.DictReader(f))
    mask_sequences.append([x['modality_mask_sha256']for x in h]);orders.append([x['loader_rng_sha256']for x in h])
   assert len(set(identities))==1,'Common modality branches did not share initialization'
   assert all(x==mask_sequences[0]for x in mask_sequences),'Modality masks were not paired'
   assert all(x==orders[0]for x in orders),'Data-order RNGs were not paired'
 for dataset in p['datasets']:
  for role in p['evaluation_roles']:
   data=evaluation_data(out,dataset,role);subsets_=subsets(data);all_models={}
   for variant in p['variants']:
    pred=[];metas=[]
    for seed in p['seeds']:
     folder=out/'evaluation'/dataset/variant/f'seed_{seed}'/role;m=read(folder/'complete.json')
     for n,h in m['files'].items():assert sha(folder/n)==h
     a=arrays(folder/'predictions.npz');assert np.array_equal(a['ids'],data['ids'])and np.array_equal(a['labels'],data['y'])
     pred.append(a);metas.append(m)
    names=pred[0]['condition_names'].tolist()
    assert all(x['condition_names'].tolist()==names for x in pred)
    margins=np.stack([x['margin']for x in pred]);eligible=np.stack([x['eligible']for x in pred])
    assert all(np.array_equal(x,eligible[0])for x in eligible)
    all_models[variant]=(margins,eligible[0],names)
    for sub,submask in subsets_.items():
     full=np.asarray([metrics(data['y'][submask],x[names.index('full'),submask])['macro_f1']for x in margins],dtype=object)
     f=[float(x)for x in full if x is not None]
     performance.append(dict(dataset=dataset,split=role,subset=sub,variant=variant,n=int(submask.sum()),training_seeds=len(p['seeds']),
      macro_f1_mean_pp=float(np.mean(f)*100)if f else None,macro_f1_sd_pp=float(np.std(f,ddof=1)*100)if len(f)>1 else None))
     families={'emap':['emap'],'cross':[n for n in names if n.startswith('cross_')],'length':[n for n in names if n.startswith('length_')],'within':[n for n in names if n.startswith('within_')]}
     for family,conditions in families.items():
      js=[names.index(n)for n in conditions];keep=submask&eligible[0,js].all(0);scores=[];baseline=[];changed=[]
      for x in margins:
       b=x[names.index('full')];bm=metrics(data['y'][keep],b[keep]);baseline.append(bm['macro_f1'])
       mm=[metrics(data['y'][keep],x[j,keep])['macro_f1']for j in js];scores.append(np.mean(mm)if all(z is not None for z in mm)and mm else None)
       changed.append(float(np.mean([(x[j,keep]>0)!=(b[keep]>0)for j in js]))if keep.any()else None)
      valid=all(z is not None for z in scores)and all(z is not None for z in baseline);delta=(np.asarray(scores,dtype=float)-np.asarray(baseline,dtype=float))*100 if valid else None
      row=dict(dataset=dataset,split=role,subset=sub,variant=variant,family=family,n=int(keep.sum()),source_videos=len(np.unique(data['source'][keep])),draws_per_seed=len(js),training_seeds=len(p['seeds']),
       full_f1_matched_mean_pp=float(np.mean(baseline)*100)if valid else None,delta_mean_pp=float(delta.mean())if valid else None,delta_sd_pp=float(delta.std(ddof=1))if valid else None,
       changed_percent=float(np.mean(changed)*100)if all(z is not None for z in changed)else None)
      # Primary intervals: official test/all; conflict intervals are secondary and pointwise.
      if valid and role in ['audit','test']and(sub in ['all','source_unseen']or(dataset=='SIMS'and sub=='nontext_support')):
       ci=paired_bootstrap(margins[:,0,:],margins[:,js,:],data['y'],data['source'],keep,p['bootstrap']['repeats'])
       assert abs(ci['delta_pp']-row['delta_mean_pp'])<1e-7
       row.update(video_ci_lo=ci['video_conditional_ci95'][0],video_ci_hi=ci['video_conditional_ci95'][1],seed_video_ci_lo=ci['video_and_seed_ci95'][0],seed_video_ci_hi=ci['video_and_seed_ci95'][1])
      effects.append(row)
   for sub,keep in subsets_.items():
    for variant in p['variants']:
     if variant=='bert_mlp':continue
     a=all_models['bert_mlp'][0][:,0,:];b=all_models[variant][0][:,0,:]
     if len(np.unique(data['y'][keep]))<2:continue
     score=np.asarray([metrics(data['y'][keep],x[keep])['macro_f1']for x in b])-np.asarray([metrics(data['y'][keep],x[keep])['macro_f1']for x in a])
     row=dict(dataset=dataset,split=role,subset=sub,variant=variant,reference='bert_mlp',n=int(keep.sum()),delta_mean_pp=float(score.mean()*100),delta_sd_pp=float(score.std(ddof=1)*100))
     if role in ['audit','test']and sub in ['all','nontext_support','source_unseen']:
      ci=paired_bootstrap(a,b,data['y'],data['source'],keep,p['bootstrap']['repeats'])
      row.update(video_ci_lo=ci['video_conditional_ci95'][0],video_ci_hi=ci['video_conditional_ci95'][1],seed_video_ci_lo=ci['video_and_seed_ci95'][0],seed_video_ci_hi=ci['video_and_seed_ci95'][1])
     comparisons.append(row)
 target=out/'review';target.mkdir(exist_ok=True)
 for name,rows in [('performance.csv',performance),('intervention_effects.csv',effects),('paired_vs_text.csv',comparisons),('model_budgets.csv',budget)]:csvout(target/name,rows)
 report=dict(status='complete',new_training_jobs=len(p['datasets'])*len(p['variants'])*len(p['seeds']),new_seeds=p['seeds'],checkpoints_sealed_before_official_evaluation=True,scientific_selection='inner dev only',historical_benchmark_tests=True,untouched_external_validation=False,
  dataset_coverage=read(out/'data_inventory.json'),parameter_and_stochastic_pairing_verified=True,emap_grouping=p['emap']['grouping'],
  uncertainty='Pointwise paired source-video intervals; secondary crossed source-video/training-seed intervals. Donor draws and anchor banks are not training seeds.',
  source_matching='Within-source is not guaranteed same speaker. Nonzero-row length is an observation proxy.',fresh_external_dataset='not installed; separate future experiment required')
 dump(target/'review_audit.json',report)
 (target/'README.md').write_text('''# KBS substantive experiment review\n\nPrimary summaries: performance.csv, intervention_effects.csv, paired_vs_text.csv. Values ending in _pp use percentage points. Both complete-input performance and intervention changes must be read. All audit/official valid/test evaluations occurred after all 75 checkpoints were selected on inner dev and sealed. These public benchmarks were previously examined and are not new untouched external test data.\n\nEMAP isolates interactions between text and grouped audio+vision. It does not isolate A-V interaction. MOSI/SIMS projections use the full empirical Cartesian product; MOSEI uses the mean of two independently sampled 64-anchor approximations. Thirty-two-anchor nested comparisons and bank margins are retained to show approximation variation.\n\nDonor draws remain nested within each training seed. Within-source effects use only eligible samples; compare with the matched full-input denominator, not the whole test set. Conflict subsets use existing SIMS unimodal labels for evaluation only. Intervals are pointwise; do not claim familywise significance or select conditions from them.\n\nRaw data, BERT files, and trained weights are not embedded in this review ZIP. Keep the server checkpoints for replay. Their SHA256 values and selected epochs are recorded.\n''',encoding='utf8')
 archive=out/'kbs_substantive_review_20261110.zip';members={}
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
  for f in out.rglob('*'):
   if not f.is_file()or f.suffix not in ['.json','.csv','.npz','.md']:continue
   rel=f.relative_to(out)
   if rel.parts[0]=='data'and f.suffix=='.npz':continue
   name=rel.as_posix();z.write(f,name);members[name]=sha(f)
  for f in [HERE/'protocol.json',HERE/'bundle_manifest.json',HERE/'PROTOCOL.md']:
   name='source/'+f.name;z.write(f,name);members[name]=sha(f)
  z.writestr('review_manifest.json',json.dumps(members,indent=2))
 with zipfile.ZipFile(archive)as z:
  assert z.testzip()is None
  for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 print('KBS_SUBSTANTIVE_REVIEW_COMPLETE',archive,flush=True)
 return archive