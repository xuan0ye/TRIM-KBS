"""CPU integration tests with synthetic data and the unmodified d=256 paper fusion."""
from pathlib import Path
import argparse,hashlib,json,sys
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import causal_common as c
import run_causal as run
from causal_common import pc,np,torch

def data():
    rng=np.random.default_rng(4);parts={};embeds={}
    for split in pc.SPLITS:
        n=8;text=np.zeros((n,3,6),dtype=np.int64)
        text[:,0]=rng.integers(1,30,(n,6));text[:,1]=1
        parts[split]=dict(ids=np.array([split+str(i%4)+'$_$'+str(i) for i in range(n)]),
            y=np.arange(n,dtype=np.int8)%2,text=text,audio=rng.normal(size=(n,6)).astype(np.float32),
            vision=rng.normal(size=(n,5)).astype(np.float32))
        embeds[split]=rng.normal(size=(n,2048)).astype(np.float32)
    return parts,embeds

def main(out):
    if out.exists():raise RuntimeError('Use a new check output directory')
    out.mkdir(parents=True);c.runtime('cpu',2);c.verify_release()
    parts,embeds=data();checks=[]
    for seed in range(10):
        ix=c.cross_video_permutation(parts['valid']['ids'],seed)
        videos=np.array([s.split('$_$')[0] for s in parts['valid']['ids']])
        assert np.all(videos!=videos[ix]) and sorted(ix.tolist())==list(range(8))
        assert np.array_equal(ix,c.cross_video_permutation(parts['valid']['ids'],seed))
    try:c.cross_video_permutation(np.array(['a$_$0','a$_$1','a$_$2','b$_$0']),1)
    except ValueError:pass
    else:raise AssertionError('Impossible donor assignment accepted')
    checks.append('label_free_cross_video_bijection_no_identity_or_same_video')
    x=np.random.default_rng(7).normal(size=(50,74));x[-10:]=0
    assert np.allclose(x.mean(0),x[np.random.default_rng(8).permutation(50)].mean(0),rtol=1e-12,atol=1e-12)
    checks.append('feature_mean_temporal_permutation_invariance_by_design')
    expected=pc.representations(parts,embeds,out/'stats');stats=pc.load_npz(out/'stats/representation_stats.npz')
    for group in pc.GROUPS:
        actual=c.build_audio(parts,embeds,stats,group)
        for s in pc.SPLITS:assert np.array_equal(actual[s]['audio'],expected[group][s]['audio'])
    checks.append('reconstruct_all_four_audio_representations_without_refitting')
    from transformers import BertConfig,BertModel
    bert=out/'bert';bert.mkdir();torch.manual_seed(42)
    cfg=BertConfig(vocab_size=32,hidden_size=768,num_hidden_layers=2,num_attention_heads=12,intermediate_size=32,max_position_embeddings=16)
    BertModel(cfg).save_pretrained(bert)
    model=c.make_model(bert,6,5,pretrained=False).eval().requires_grad_(False)
    original_hash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    z={}
    for s in pc.SPLITS:
        z[s]=c.encode_cache(model,parts[s],out/(s+'_cache.npz'),dict(synthetic=True,split=s),torch.device('cpu'))
        repeated=c.encode_cache(model,parts[s],out/(s+'_cache.npz'),dict(synthetic=True,split=s),torch.device('cpu'))
        for key in z[s]:assert np.array_equal(z[s][key],repeated[key])
    tb=torch.as_tensor(parts['valid']['text']);ta=torch.as_tensor(parts['valid']['audio']);tv=torch.as_tensor(parts['valid']['vision'])
    ix=c.cross_video_permutation(parts['valid']['ids'],3)
    with torch.inference_mode():
        direct,_=model(tb[:,0],tb[:,1],tb[:,2],ta,tv,training=False)
        _,shuffled_a,_=c.encode(model,tb,ta[ix],tv)
    assert np.max(np.abs(direct.numpy()-z['valid']['logits_TAV']))<1e-5
    assert np.array_equal(shuffled_a.numpy(),z['valid']['ha'][ix])
    checks.append('legacy_forward_and_cached_projection_shuffle_parity')
    head=c.EventResidual(len(c.EVENTS));base=torch.randn(8,2);ht=torch.randn(8,256);e=torch.rand(8,len(c.EVENTS))
    assert torch.equal(head(base,ht,e,torch.ones(8))[0],base)
    with torch.no_grad():head.delta[-1].weight.fill_(.3);head.delta[-1].bias.fill_(.4)
    absent=head(base,ht,e,torch.zeros(8))[0];assert torch.equal(absent,base)
    assert head(base,ht,e,torch.ones(8))[1].abs().max()<=2
    optimizer=torch.optim.AdamW(head.parameters(),lr=.01);old=head.delta[-1].weight.detach().clone()
    loss=torch.nn.functional.cross_entropy(head(base,ht,e,torch.ones(8))[0],torch.arange(8)%2)
    optimizer.zero_grad();loss.backward();optimizer.step()
    assert not torch.equal(old,head.delta[-1].weight)
    checks.append('identity_initialization_missing_audio_invariance_bounded_correction_and_real_gradient')
    output=out/'pipeline';output.mkdir();pc.dump_json(output/'pipeline_signature.json',dict(synthetic=True))
    args=SimpleNamespace(device=torch.device('cpu'),fusion_batch=4,permutations=3,epochs=3,head_batch=4,out=output)
    run.diagnostic(model,z['valid'],'synthetic_paper',output,args)
    fake=out/'fake_parent';cache=fake/'feature_cache';cache.mkdir(parents=True);records=[]
    classrows=list(__import__('csv').DictReader((c.HERE.parent/'audio_event_remote_20261024/assets/class_labels_indices.csv').open(encoding='utf-8-sig')))
    names=[r['display_name'] for r in classrows]
    for s in pc.SPLITS:
        for i,sid in enumerate(parts[s]['ids']):
            events=np.zeros(527,dtype=np.float32);events[names.index('Music')]=.8 if i%2 else .01
            source=hashlib.sha256(str(sid).encode()).hexdigest()
            path=cache/(source+'.npz');pc.atomic_npz(path,events=events,source_sha256=np.array(source))
            records.append(dict(id=str(sid),split=s,source_sha256=source,cache_sha256=pc.sha_file(path),
                                relative_path=pc.media_rel(sid),seconds=2.))
    pc.dump_json(output/'media_manifest.json',dict(records=records))
    event_dir=c.HERE.parent/'audio_event_remote_20261024'
    with patch.object(run,'PARENT',fake):
        events,eligible=run.collect_events(output,event_dir,parts,output)
    assert not eligible['eligible'] and eligible['train_candidates_ge_0_3']==4
    candidates=list(__import__('csv').DictReader((output/'listening_candidates_train.csv').open(encoding='utf-8-sig')))
    assert all(row['id'] in parts['train']['ids'] for row in candidates)
    checks.append('canonical_event_cache_read_train_only_candidates_and_insufficient_coverage_screen')
    for kind in ('correct_events','shuffled_events','zero_events'):
        result=run.fit_head(z['train'],z['valid'],events,output,kind,args)
        assert result['missing_audio_unchanged'] and not result['test_evaluated']
        again=run.fit_head(z['train'],z['valid'],events,output,kind,args);assert again==result
    checks.append('three_matched_capacity_heads_training_selection_missing_audio_invariance_and_completed_cache')
    # A known informative event must be learnable, while the identical zero-event
    # head cannot solve balanced labels from constant text and constant parent logits.
    informative=out/'informative';informative.mkdir()
    pc.dump_json(informative/'pipeline_signature.json',dict(synthetic_informative_event=True))
    toy={};toy_events={}
    for split,n in (('train',128),('valid',48)):
        y=np.arange(n,dtype=np.int8)%2
        toy[split]=dict(ids=np.array([split+str(i//4)+'$_$'+str(i) for i in range(n)]),y=y,ht=np.zeros((n,256),dtype=np.float32))
        for mode in c.MODES:toy[split]['logits_'+mode]=np.zeros((n,2),dtype=np.float32)
        toy_events[split]=np.zeros((n,len(c.EVENTS)),dtype=np.float32);toy_events[split][:,0]=.1+.8*y
    toy_args=SimpleNamespace(device=torch.device('cpu'),permutations=3,epochs=40,head_batch=16)
    learned=run.fit_head(toy['train'],toy['valid'],toy_events,informative,'correct_events',toy_args)
    placebo=run.fit_head(toy['train'],toy['valid'],toy_events,informative,'zero_events',toy_args)
    assert learned['TAV_macro_f1']>.9 and learned['TAV_macro_f1']>placebo['TAV_macro_f1']+.4
    assert learned['event_assignment_vs_extra_shuffle_gain_pp']>20
    checks.append('informative_synthetic_event_learned_with_gain_over_zero_and_same_checkpoint_shuffle')
    assert original_hash==hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    checks.append('original_backbone_parameters_unchanged')
    run.package(output,dict(status='complete',synthetic=True,test_evaluated=False))
    import zipfile
    with zipfile.ZipFile(output/'audio_causal_review_20261026.zip') as archive:
        assert archive.testzip() is None
        manifest=json.loads(archive.read('audio_causal_review/review_manifest.json'))
        for rel,digest in manifest['files'].items():
            assert hashlib.sha256(archive.read('audio_causal_review/'+rel)).hexdigest()==digest,rel
    checks.append('review_archive_all_hashes_and_metric_prediction_exports')
    report=dict(status='passed',synthetic=True,test_evaluated=False,checks=checks,torch=torch.__version__,
        source_hashes={f:pc.sha_file(c.HERE/f) for f in ('causal_common.py','run_causal.py','remote_audio_causal.sh')},
        note='Synthetic 2-layer BERT with unchanged d=256 paper fusion. No claimed experimental improvement.')
    pc.dump_json(out/'engineering_check.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();main(a.out)
