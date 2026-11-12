"""Recompute validation metrics and export a small review ZIP without model weights."""
import argparse,zipfile
from paper_common import *

def summarize(out):
    rows=[];jobs={};paired={};identities={}
    for group in GROUPS:
        folders=sorted((out/group).glob('seed_*'))
        if not folders: raise RuntimeError(f'Missing training group: {group}')
        for folder in folders:
            complete=read_json(folder/'complete.json')
            if complete['test_evaluated'] or complete['status']!='complete': raise RuntimeError('Invalid completion record')
            for f,digest in complete['artifacts'].items():
                if sha_file(folder/f)!=digest: raise RuntimeError(f'Artifact changed: {folder/f}')
            seed=complete['signature']['seed'];ids=load_npz(folder/'validation_predictions.npz')
            key=str(seed);identities.setdefault(key,complete['identity'])
            if identities[key]!=complete['identity']: raise RuntimeError('Groups did not share initialization/parameter count')
            ref=read_json(folder/'validation_metrics.json')['metrics']
            for condition in ref:
                prob=ids['prob_'+condition];pred=ids['pred_'+condition]
                m=scores(ids['labels'],pred,prob)
                for k,v in m.items():
                    if abs(v-ref[condition][k])>1e-7: raise RuntimeError(f'Metric replay mismatch: {condition}/{k}')
                rows.append(dict(group=group,seed=seed,condition=condition,**m))
            jobs[(group,seed)]=(ids,complete)
    seeds=sorted(s for g,s in jobs if g==GROUPS[0])
    if any(sorted(s for g,s in jobs if g==group)!=seeds for group in GROUPS): raise RuntimeError('Seed coverage differs')
    comparisons=[]
    for seed in seeds:
        base=jobs[('paper_original',seed)][0]
        for group in GROUPS[1:]:
            current=jobs[(group,seed)][0]
            if not np.array_equal(base['ids'],current['ids']) or not np.array_equal(base['labels'],current['labels']): raise RuntimeError('Validation rows differ')
            for condition in MASKS:
                b=base['pred_'+condition];c=current['pred_'+condition];y=base['labels']
                bm=scores(y,b);cm=scores(y,c)
                comparisons.append(dict(group=group,reference='paper_original',seed=seed,condition=condition,
                    delta_macro_f1_pp=100*(cm['macro_f1']-bm['macro_f1']),delta_accuracy_pp=100*(cm['accuracy']-bm['accuracy']),
                    corrected=int(np.sum((b!=y)&(c==y))),broken=int(np.sum((b==y)&(c!=y))),prediction_changes=int(np.sum(b!=c))))
    write_csv(out/'validation_summary.csv',rows);write_csv(out/'paired_changes.csv',comparisons)
    plan=read_json(out/'inventory.json')
    dump_json(out/'review_summary.json',dict(status='complete',scope='available_media_subset_validation_only',smoke=plan['smoke'],
        seeds=seeds,availability=plan['availability'],counts={s:len(plan['splits'][s]) for s in SPLITS},
        initialization=identities,test_evaluated=False,claim='Pilot estimates only; no full-dataset or test-set improvement claim',
        audio_representation='PANNs 2048 -> fixed Gaussian projection to original audio dimension, training-only scaling',
        selection='global TAV validation Macro-F1; other conditions are secondary outcomes',
        limitations=['Raw-media availability may select a biased subset','AudioSet pretraining is an additional external resource',
                     'Fixed projection may discard information','Validation is used for both checkpoint selection and exploratory comparison',
                     'Independent full-model training can change T and TV performance']))
    files=[out/f for f in ['inventory.json','excluded_media.csv','bert_verified.json','extraction_signature.json','media_manifest.json',
           'extraction_progress.json','representation_stats.npz','suite_progress.json','validation_summary.csv','paired_changes.csv','review_summary.json']]
    for (group,seed),(data,complete) in jobs.items():
        folder=out/group/f'seed_{seed}'
        files.extend(folder/f for f in ['complete.json','best_validation.json','model_identity.json','validation_history.csv','validation_predictions.npz','validation_metrics.json'])
    bundle=out/'paper_audio_review_20261025.zip';tmp=bundle.with_suffix('.tmp.zip')
    manifest={str(p.relative_to(out)).replace('\\','/'):sha_file(p) for p in files}
    manifest['EXPERIMENT_PROTOCOL.md']=sha_file(HERE/'EXPERIMENT_PROTOCOL.md')
    manifest['source_bundle_manifest.json']=sha_file(HERE/'bundle_manifest.json')
    dump_json(out/'review_manifest.json',dict(files=manifest,test_evaluated=False))
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[out/'review_manifest.json']: z.write(path,'paper_audio_review/'+str(path.relative_to(out)).replace('\\','/'))
        z.write(HERE/'EXPERIMENT_PROTOCOL.md','paper_audio_review/EXPERIMENT_PROTOCOL.md')
        z.write(HERE/'bundle_manifest.json','paper_audio_review/source_bundle_manifest.json')
    os.replace(tmp,bundle)
    print('PAPER_AUDIO_REVIEW_COMPLETE',bundle,'test_evaluated=False',flush=True)
    return bundle

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();verify_code();summarize(a.out)
