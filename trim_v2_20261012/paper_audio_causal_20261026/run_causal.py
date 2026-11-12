"""Fixed-checkpoint audio interventions, cached event census, and three controlled residual pilots."""
import argparse,csv,gc,time,zipfile
from causal_common import *
GROUPS=pc.GROUPS

def score(y,logits,base_pred=None):
    pred=logits.argmax(1);prob=probabilities(logits)
    row=pc.scores(y,pred,prob)
    if base_pred is not None:
        row.update(prediction_changes=int((pred!=base_pred).sum()),
                   corrected=int(((pred==y)&(base_pred!=y)).sum()),broken=int(((pred!=y)&(base_pred==y)).sum()))
    return row

def diagnostic(model,z,group,out,args):
    folder=out/'diagnostics'/group;folder.mkdir(parents=True,exist_ok=True)
    predictions=dict(ids=z['ids'],labels=z['y']);rows=[];baseline=z['logits_TAV'].argmax(1)
    reference=None;baseline_prob=probabilities(z['logits_TAV'])
    def add(tag,logits,kind,rep=-1,donors=None):
        row=dict(group=group,condition=tag,kind=kind,rep=rep,**score(z['y'],logits,baseline))
        row['mean_abs_probability_change']=float(np.abs(probabilities(logits)-baseline_prob).mean())
        rows.append(row);predictions['prob_'+tag]=probabilities(logits);predictions['pred_'+tag]=logits.argmax(1)
        if donors is not None:predictions['donor_'+tag]=donors
    for c in MODES:add(c,z['logits_'+c],'original_or_missing')
    original,prob=fusion_predict(model,z,args.device,batch=args.fusion_batch)
    if not np.array_equal(original.argmax(1),baseline) or np.max(np.abs(prob-baseline_prob))>1e-5:
        raise RuntimeError('Cached fusion replay differs from the original parent predictions')
    zero,_=fusion_predict(model,z,args.device,zero=True,batch=args.fusion_batch)
    add('zero_audio_present',zero,'zero_present')
    for rep in range(args.permutations):
        for kind in ('audio_random','audio_cross_video','visual_cross_video'):
            seed=20261026+rep
            ix=pc.derangement(len(z['y']),seed) if kind=='audio_random' else cross_video_permutation(z['ids'],seed)
            logits,_=fusion_predict(model,z,args.device,
                a_ix=ix if kind.startswith('audio_') else None,v_ix=ix if kind.startswith('visual_') else None,
                batch=args.fusion_batch)
            add(kind+'_'+str(rep).zfill(2),logits,kind,rep,ix)
    pc.atomic_npz(folder/'validation_predictions.npz',**predictions);pc.write_csv(folder/'metrics.csv',rows)
    summary=[]
    for kind in ('audio_random','audio_cross_video','visual_cross_video'):
        selected=[r for r in rows if r['kind']==kind]
        summary.append(dict(group=group,intervention=kind,repetitions=len(selected),
            original_macro_f1=rows[3]['macro_f1'],permuted_macro_f1_mean=float(np.mean([r['macro_f1'] for r in selected])),
            correspondence_gain_pp=100*(rows[3]['macro_f1']-float(np.mean([r['macro_f1'] for r in selected]))),
            permuted_macro_f1_sd=float(np.std([r['macro_f1'] for r in selected])),
            mean_prediction_changes=float(np.mean([r['prediction_changes'] for r in selected])),
            mean_abs_probability_change=float(np.mean([r['mean_abs_probability_change'] for r in selected]))))
    pc.dump_json(folder/'summary.json',dict(summary=summary,test_evaluated=False,checkpoint_fixed=True))
    print('CAUSAL_DIAGNOSTIC_COMPLETE',group,json.dumps(summary),flush=True)
    return summary

def collect_events(run,event_dir,parts,out):
    labelpath=event_dir/'assets/class_labels_indices.csv'
    if pc.sha_file(labelpath)!=LABEL_SHA:raise RuntimeError('AudioSet class order differs from pinned metadata')
    with labelpath.open(encoding='utf-8-sig',newline='') as f:labels=[r['display_name'] for r in csv.DictReader(f)]
    if len(labels)!=527 or any(e not in labels for e in EVENTS):raise RuntimeError('Missing pinned environment event classes')
    columns=[labels.index(e) for e in EVENTS]
    media=pc.read_json(run/'media_manifest.json')
    lookup={r['id']:r for r in media['records']}
    features={};coverage=[];candidates=[];train_candidates=0;candidate_videos=set();done=0
    for split in pc.SPLITS:
        rows=[]
        for sid in parts[split]['ids']:
            record=lookup[str(sid)]
            path=PARENT/'feature_cache'/(hashlib.sha256(str(sid).encode()).hexdigest()+'.npz')
            if pc.sha_file(path)!=record['cache_sha256']:raise RuntimeError(f'PANNs cache changed: {sid}')
            item=pc.load_npz(path)
            if str(item['source_sha256'].item())!=record['source_sha256']:raise RuntimeError(f'Cache/media ID mismatch: {sid}')
            e=item['events']
            if e.shape!=(527,) or not np.isfinite(e).all() or np.any((e<0)|(e>1)):raise RuntimeError('Invalid event probabilities')
            rows.append(e[columns]);done+=1
            if done%500==0:print('EVENT_CACHE_READ',done,flush=True)
        features[split]=np.stack(rows)
        for j,event in enumerate(EVENTS):
            values=features[split][:,j]
            coverage.append(dict(split=split,event=event,n=len(values),mean=float(values.mean()),maximum=float(values.max()),
                count_ge_0_1=int((values>=.1).sum()),count_ge_0_3=int((values>=.3).sum()),count_ge_0_5=int((values>=.5).sum())))
        hits=features[split].max(1)>=.3
        if split=='train':
            train_candidates=int(hits.sum())
            candidate_videos={str(x).split('$_$')[0] for x in parts[split]['ids'][hits]}
            # Targeted candidates, chosen only from TRAIN, with source-video diversity.
            selected=set()
            for j,event in enumerate(EVENTS):
                used_videos=set();count=0
                for i in np.argsort(features[split][:,j])[::-1]:
                    if features[split][i,j]<.1 or count==3:break
                    sid=str(parts[split]['ids'][i]);video=sid.split('$_$')[0]
                    if video in used_videos:continue
                    used_videos.add(video);selected.add(sid);count+=1
                    r=lookup[sid]
                    candidates.append(dict(selection='event_candidate',event=event,score=float(features[split][i,j]),id=sid,
                        relative_path=r['relative_path'],seconds=r['seconds'],human_verified=False))
            for i in np.random.default_rng(20261026).permutation(len(parts[split]['ids'])):
                sid=str(parts[split]['ids'][i])
                if sid in selected:continue
                r=lookup[sid]
                candidates.append(dict(selection='random_control',event='',score='',id=sid,relative_path=r['relative_path'],
                                       seconds=r['seconds'],human_verified=False))
                if sum(r['selection']=='random_control' for r in candidates)==10:break
    pc.write_csv(out/'event_coverage.csv',coverage);pc.write_csv(out/'listening_candidates_train.csv',candidates)
    pc.atomic_npz(out/'selected_event_features.npz',event_names=np.array(EVENTS),
        **{split+'_ids':parts[split]['ids'] for split in pc.SPLITS},
        **{split+'_events':features[split] for split in pc.SPLITS})
    eligibility=dict(train_candidates_ge_0_3=train_candidates,train_candidate_source_videos=len(candidate_videos),
        train_threshold=100,source_video_threshold=20,eligible=train_candidates>=100 and len(candidate_videos)>=20,
        note='Detector-based feasibility screen, not human event truth. No validation labels used in eligibility.')
    pc.dump_json(out/'event_eligibility.json',eligibility)
    print('EVENT_CANDIDATE_COVERAGE',json.dumps(eligibility),flush=True)
    return features,eligibility

def head_predict(head,z,features,mode,device,batch=256):
    output=[]
    with torch.inference_mode():
        for start in range(0,len(z['y']),batch):
            sl=slice(start,start+batch)
            logits,_=head(torch.as_tensor(z['logits_'+mode][sl],device=device),torch.as_tensor(z['ht'][sl],device=device),
                          torch.as_tensor(features[sl],device=device),
                          torch.full((len(z['y'][sl]),),BITS[mode][1],device=device))
            output.append(logits.cpu().numpy())
    return np.concatenate(output)

def fit_head(train,valid,events,out,kind,args):
    folder=out/'event_pilot'/kind;folder.mkdir(parents=True,exist_ok=True)
    signature=dict(parent_signature=pc.read_json(out/'pipeline_signature.json'),kind=kind,epochs=args.epochs,seed=42,lr=.001,
                   event_names=EVENTS,head='bounded_gated_event_residual',train_modality_dropout=.15,
                   parent_frozen=True,selection='validation_TAV_macro_f1_including_identity_epoch_0')
    signature=json.loads(json.dumps(signature))
    complete=folder/'complete.json'
    if complete.exists():
        old=pc.read_json(complete)
        if old['signature']!=signature:raise RuntimeError('Residual completed configuration changed')
        for rel,digest in old['artifacts'].items():
            if pc.sha_file(folder/rel)!=digest:raise RuntimeError('Residual artifact changed')
        return old['summary']
    features={s:e.copy() for s,e in events.items()}
    if kind=='shuffled_events':
        for i,s in enumerate(pc.SPLITS):features[s]=features[s][cross_video_permutation(train['ids'] if s=='train' else valid['ids'],20261027+i)]
    elif kind=='zero_events':features={s:np.zeros_like(e) for s,e in features.items()}
    torch.manual_seed(42);head=EventResidual(len(EVENTS)).to(args.device)
    initialization=hashlib.sha256(b''.join(x.detach().cpu().numpy().tobytes() for x in head.state_dict().values())).hexdigest()
    pc.record_json(out/'event_head_initialization.json',dict(sha256=initialization,parameters=sum(p.numel() for p in head.parameters())))
    # Identity initialization must reproduce the unchanged parent for ALL missingness conditions.
    for mode in MODES:
        assert np.array_equal(head_predict(head,valid,features['valid'],mode,args.device),valid['logits_'+mode])
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.001)
    counts=np.bincount(train['y'],minlength=2);w=np.sqrt(counts.sum()/(2*counts));w=w/w.mean()
    ce=torch.nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=args.device),label_smoothing=.05)
    rng=np.random.default_rng(42);mask_rng=torch.Generator().manual_seed(42)
    best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    best_score=score(valid['y'],valid['logits_TAV'])['macro_f1'];best_epoch=0
    history=[dict(epoch=0,train_loss=None,val_macro_f1=best_score)]
    stacked=np.stack([train['logits_'+m] for m in MODES],axis=1)
    for epoch in range(1,args.epochs+1):
        head.train();losses=[]
        for start in range(0,len(train['y']),args.head_batch):
            # One shared permutation per epoch, with independent paired mask RNG.
            if start==0:order=rng.permutation(len(train['y']))
            ix=order[start:start+args.head_batch];n=len(ix)
            available=(torch.rand((n,2),generator=mask_rng)>=.15).numpy()
            condition=available[:,0].astype(int)+2*available[:,1].astype(int)
            base=torch.as_tensor(stacked[ix,condition],device=args.device)
            optimizer.zero_grad(set_to_none=True)
            logits,d=head(base,torch.as_tensor(train['ht'][ix],device=args.device),
                          torch.as_tensor(features['train'][ix],device=args.device),
                          torch.as_tensor(available[:,0],device=args.device))
            loss=ce(logits,torch.as_tensor(train['y'][ix],device=args.device).long())+.001*d.square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite event-head loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
        head.eval();logits=head_predict(head,valid,features['valid'],'TAV',args.device)
        f1=score(valid['y'],logits)['macro_f1']
        if f1>best_score:
            best_score=f1;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),val_macro_f1=f1))
        pc.dump_json(out/'current_job.json',dict(stage='event_pilot',kind=kind,epoch=epoch,epochs=args.epochs,best=best_score))
        print('EVENT_HEAD_EPOCH',kind,epoch,'val_f1',round(f1,6),'best',round(best_score,6),flush=True)
    head.load_state_dict(best_state);head.eval();torch.save(best_state,folder/'best_head.pt')
    arrays=dict(ids=valid['ids'],labels=valid['y']);rows=[];baseline=valid['logits_TAV'].argmax(1)
    for mode in MODES:
        logits=head_predict(head,valid,features['valid'],mode,args.device)
        if mode in ('T','TV') and not np.array_equal(logits,valid['logits_'+mode]):
            raise RuntimeError('Event branch changed predictions when audio is absent')
        arrays['pred_'+mode]=logits.argmax(1);arrays['prob_'+mode]=probabilities(logits)
        rows.append(dict(kind=kind,condition=mode,**score(valid['y'],logits,baseline)))
    if abs(rows[3]['macro_f1']-best_score)>1e-12:raise RuntimeError('Selected event-head checkpoint replay mismatch')
    shuffles=[]
    for rep in range(args.permutations):
        ix=cross_video_permutation(valid['ids'],20262026+rep)
        logits=head_predict(head,valid,features['valid'][ix],'TAV',args.device)
        row=score(valid['y'],logits,arrays['pred_TAV']);shuffles.append(row['macro_f1'])
        arrays['shuffle_pred_'+str(rep)]=logits.argmax(1);arrays['shuffle_prob_'+str(rep)]=probabilities(logits)
    pc.atomic_npz(folder/'validation_predictions.npz',**arrays);pc.write_csv(folder/'metrics.csv',rows)
    pc.dump_json(folder/'history.json',history)
    summary=dict(kind=kind,best_epoch=best_epoch,baseline_TAV_macro_f1=score(valid['y'],valid['logits_TAV'])['macro_f1'],
        TAV_macro_f1=best_score,TV_macro_f1=score(valid['y'],valid['logits_TV'])['macro_f1'],
        event_assignment_vs_extra_shuffle_gain_pp=100*(best_score-float(np.mean(shuffles))),
        extra_shuffle_macro_f1_mean=float(np.mean(shuffles)),extra_shuffle_macro_f1_sd=float(np.std(shuffles)),
        missing_audio_unchanged=True,head_parameters=sum(p.numel() for p in head.parameters()),test_evaluated=False)
    artifacts={f:pc.sha_file(folder/f) for f in ('best_head.pt','validation_predictions.npz','metrics.csv','history.json')}
    pc.dump_json(complete,dict(signature=signature,summary=summary,artifacts=artifacts,status='complete',test_evaluated=False))
    print('EVENT_HEAD_COMPLETE',json.dumps(summary),flush=True)
    return summary

def package(out,summary):
    pc.dump_json(out/'summary.json',summary)
    files=[p for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.npz') and 'cache' not in p.relative_to(out).parts
           and p.name!='review_manifest.json']
    manifest={p.relative_to(out).as_posix():pc.sha_file(p) for p in files}
    manifest['PROTOCOL.md']=pc.sha_file(HERE/'PROTOCOL.md')
    manifest['source_manifest.json']=pc.sha_file(HERE/'bundle_manifest.json')
    pc.dump_json(out/'review_manifest.json',dict(files=manifest,test_evaluated=False))
    target=out/'audio_causal_review_20261026.zip';temp=target.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[out/'review_manifest.json']:z.write(p,'audio_causal_review/'+p.relative_to(out).as_posix())
        z.write(HERE/'PROTOCOL.md','audio_causal_review/PROTOCOL.md')
        z.write(HERE/'bundle_manifest.json','audio_causal_review/source_manifest.json')
    os.replace(temp,target);print('AUDIO_CAUSAL_REVIEW_COMPLETE',target,flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=HERE.parent.parent);ap.add_argument('--run',type=Path,default=PARENT/'runs/pilot')
    ap.add_argument('--event-dir',type=Path,default=HERE.parent/'audio_event_remote_20261024')
    ap.add_argument('--out',type=Path,default=HERE/'runs/pilot');ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--permutations',type=int,default=20);ap.add_argument('--epochs',type=int,default=15)
    ap.add_argument('--fusion-batch',type=int,default=64);ap.add_argument('--head-batch',type=int,default=32)
    ap.add_argument('--diagnostics-only',action='store_true');args=ap.parse_args()
    if args.permutations<2 or args.epochs<1:raise ValueError('Need >=2 permutations and >=1 epoch')
    release=verify_release();args.out.mkdir(parents=True,exist_ok=True);args.device=runtime(args.device,4)
    if args.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    inputs=verify_parent(args.run)
    pc.record_json(args.out/'pipeline_signature.json',dict(release=release,parent_inputs=inputs,permutations=args.permutations,
        epochs=args.epochs,head_batch=args.head_batch,fusion_batch=args.fusion_batch,device_type=args.device.type,
        diagnostics_only=args.diagnostics_only,event_names=list(EVENTS),torch=torch.__version__,test_evaluated=False))
    from paper_data import load_parts
    parts,embeds=load_parts(args.run);stats=pc.load_npz(args.run/'representation_stats.npz')
    events,eligibility=collect_events(args.run,args.event_dir,parts,args.out)
    all_summary=[];original_train=None;original_valid=None
    for group in GROUPS:
        data=build_audio(parts,embeds,stats,group)
        model=load_backbone(args.root,args.run,group,data['train']['audio'].shape[1],data['train']['vision'].shape[1],args.device)
        cache=args.out/'cache';cache.mkdir(exist_ok=True)
        valid=encode_cache(model,data['valid'],cache/(group+'_valid.npz'),feature_signature(args.run,group,'valid'),args.device)
        reference=pc.load_npz(args.run/group/'seed_42/validation_predictions.npz')
        for c in MODES:
            logits=valid['logits_'+c]
            if not np.array_equal(reference['ids'],valid['ids']) or not np.array_equal(reference['pred_'+c],logits.argmax(1)):
                raise RuntimeError('Parent validation ID/prediction replay mismatch')
            if np.max(np.abs(probabilities(logits)-reference['prob_'+c]))>1e-5:raise RuntimeError('Parent probability replay mismatch')
        all_summary+=diagnostic(model,valid,group,args.out,args)
        if group=='paper_original':
            original_valid=valid
            if eligibility['eligible'] and not args.diagnostics_only:
                original_train=encode_cache(model,data['train'],cache/'paper_original_train.npz',
                                           feature_signature(args.run,group,'train'),args.device)
        del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    pc.write_csv(args.out/'diagnostic_summary.csv',all_summary)
    event_results=[]
    if original_train is not None:
        for kind in ('correct_events','shuffled_events','zero_events'):
            event_results.append(fit_head(original_train,original_valid,events,args.out,kind,args))
    event_status='complete' if event_results else 'skipped_diagnostics_only' if args.diagnostics_only else 'skipped_insufficient_training_candidates'
    summary=dict(status='complete',diagnostic_groups=list(GROUPS),diagnostic_summary=all_summary,event_pilot_status=event_status,
                 event_eligibility=eligibility,event_pilot_results=event_results,train_n=len(parts['train']['y']),valid_n=len(parts['valid']['y']),
                 model_scope='Four completed TRIM representation variants only; no external baseline executed.',
                 order_scope='Across-sample input correspondence only. Feature-frame means are permutation invariant by construction. Raw waveform order not tested.',
                 evidence_scope='Exploratory selected-validation pilot; no new test evaluation, no universal or novel finding claimed.',test_evaluated=False)
    package(args.out,summary);print('AUDIO_CAUSAL_SUITE_COMPLETE test_evaluated=False',flush=True)
if __name__=='__main__':main()
