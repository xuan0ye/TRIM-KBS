"""Meaningful CPU checks: data controls, legacy parity, full stage chain and exact resume."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paper_common import *
import torch
from types import SimpleNamespace
from unittest.mock import patch
import paper_train as trainer
from paper_data import load_parts
from paper_summarize import summarize

def synthetic(seed=4):
    rng=np.random.default_rng(seed);parts={};embeds={}
    for s,n in [('train',8),('valid',8)]:
        text=np.zeros((n,3,6),dtype=np.int64);text[:,0]=rng.integers(1,30,(n,6));text[:,1]=1
        parts[s]=dict(ids=np.array([s+str(i) for i in range(n)]),y=np.arange(n,dtype=np.int8)%2,
                      text=text,audio=rng.normal(size=(n,6)).astype(np.float32),vision=rng.normal(size=(n,5)).astype(np.float32))
        embeds[s]=rng.normal(size=(n,2048)).astype(np.float32)
    return parts,embeds

def run(out):
    if out.exists(): raise RuntimeError('Use a new check output directory')
    out.mkdir(parents=True);trainer.runtime('cpu',2);torch.manual_seed(123)
    from transformers import BertConfig,BertModel
    bert=out/'bert';bert.mkdir()
    config=BertConfig(vocab_size=32,hidden_size=768,num_hidden_layers=2,num_attention_heads=12,intermediate_size=32,max_position_embeddings=16)
    BertModel(config).save_pretrained(bert)
    parts,embeds=synthetic();groups=representations(parts,embeds,out)
    assert np.array_equal(groups['paper_original']['train']['audio'],parts['train']['audio'])
    assert np.max(np.abs(groups['paper_normalized']['train']['audio'].mean(0)))<1e-6
    a=groups['panns_projected']['train']['audio'];b=groups['panns_shuffled']['train']['audio']
    assert a.shape==b.shape==parts['train']['audio'].shape and not np.any(np.all(a==b,axis=1))
    changed={s:dict(parts[s]) for s in SPLITS};changed['valid']['audio']=parts['valid']['audio']+10000
    changed_emb={s:v.copy() for s,v in embeds.items()};changed_emb['valid']*=999
    second=representations(changed,changed_emb,out/'stats_variant')
    for g in GROUPS: assert np.array_equal(second[g]['train']['audio'],groups[g]['train']['audio'])
    x=load_npz(out/'representation_stats.npz');z=load_npz(out/'stats_variant/representation_stats.npz')
    for k in x: assert np.array_equal(x[k],z[k]),k
    # An explicitly misordered feature table must fail, rather than silently join by row.
    fake=out/'misaligned';fake.mkdir()
    dump_json(fake/'inventory.json',dict(splits={s:[dict(id=i,label=int(y)) for i,y in zip(parts[s]['ids'],parts[s]['y'])] for s in SPLITS}))
    for s in SPLITS:
        atomic_npz(fake/f'paper_{s}.npz',**parts[s]);atomic_npz(fake/f'embedding_{s}.npz',ids=parts[s]['ids'][::-1],embedding=embeds[s])
    try: load_parts(fake)
    except RuntimeError: pass
    else: raise AssertionError('Misaligned IDs accepted')
    plan=dict(smoke=True,splits={s:[dict(id=i,label=int(y)) for i,y in zip(parts[s]['ids'],parts[s]['y'])] for s in SPLITS},availability={s:dict(available=8) for s in SPLITS})
    dump_json(out/'inventory.json',plan)
    for f in ('bert_verified.json','extraction_signature.json','media_manifest.json','extraction_progress.json','suite_progress.json'): dump_json(out/f,dict(synthetic=True,test_evaluated=False))
    write_csv(out/'excluded_media.csv',[dict(split='train',id='none',reason='synthetic')])
    args=SimpleNamespace(out=out,smoke=True,batch=4,acc_steps=1,device=torch.device('cpu'))
    inputs=dict(synthetic=True)
    for g in GROUPS: trainer.run_job(args,g,42,groups[g],bert,inputs)
    summary=summarize(out)
    assert summary.is_file()
    original=out/'paper_original/seed_42'
    before=read_json(original/'complete.json');trainer.run_job(args,'paper_original',42,groups['paper_original'],bert,inputs)
    assert before==read_json(original/'complete.json')
    # Interrupt after the first epoch's atomic RNG/optimizer checkpoint, then resume.
    resumed=out/'resumed';args.out=resumed
    real_save=trainer.save_torch;interrupted=[False]
    def stop(path,value):
        real_save(path,value)
        if Path(path).name=='resume.pt' and not interrupted[0]:
            interrupted[0]=True;raise InterruptedError('Injected interruption after saved epoch')
    try:
        with patch.object(trainer,'save_torch',side_effect=stop): trainer.run_job(args,'paper_original',42,groups['paper_original'],bert,inputs)
    except InterruptedError: pass
    else: raise AssertionError('Interruption not triggered')
    trainer.run_job(args,'paper_original',42,groups['paper_original'],bert,inputs)
    recovered=resumed/'paper_original/seed_42'
    p=load_npz(original/'validation_predictions.npz');q=load_npz(recovered/'validation_predictions.npz')
    for k in p: assert np.array_equal(p[k],q[k]),('Resume differs',k)
    first=torch.load(original/'best.pt',weights_only=True,map_location='cpu');second=torch.load(recovered/'best.pt',weights_only=True,map_location='cpu')
    for k in first: assert torch.equal(first[k],second[k]),('Resume parameter differs',k)
    manifest=read_json(HERE/'bundle_manifest.json')
    assert manifest['files']['_paper/model_legacy.py']==sha_file(HERE.parent.parent/'audit_av_core_20261012/model_legacy.py')
    report=dict(status='passed',checks=['train_only_statistics','same_audio_dimension','fixed_derangement','reject_misaligned_ids',
                 'four_group_three_phase_cpu_training','same_parameter_count_and_initialization','legacy_forward_parity',
                 'absent_audio_input_invariance','validation_metric_replay','review_zip_export','completed_cache_verification','exact_epoch_resume_weights_and_predictions'],
                synthetic=True,test_evaluated=False,torch=torch.__version__,note='Synthetic 2-layer BERT; real paper fusion d=256. This is not an experimental improvement result.')
    dump_json(out/'engineering_check.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run(a.out)
