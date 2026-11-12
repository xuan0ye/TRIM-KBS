"""Validation-only supervised full-PANNs residual on a fixed paper TRIM checkpoint."""
import argparse,time,zipfile
from fusion_core import *

def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def save_weights(path,weights):
 tmp=path.with_suffix('.tmp');torch.save(weights,tmp);os.replace(tmp,path)

def fingerprint(model):
 return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()

def assigned_features(data,kind,donors):
 if kind=='zero_supervised':return {s:np.zeros_like(data[s]['x']) for s in ('train','valid')}
 return {s:data[s]['x'][donors[s]] if kind=='shuffled_supervised' else data[s]['x'] for s in ('train','valid')}

def training_plan(n,epochs,seed):
 rng=np.random.default_rng(seed);order=[];a=[];v=[]
 for epoch in range(epochs):
  order.append(rng.permutation(n));a.append(rng.random(n)>=.15);v.append(rng.random(n)>=.15)
 return dict(order=np.stack(order),audio_present=np.stack(a),vision_present=np.stack(v))

def fit(args,data,contexts,kind,seed,donors,warm_state,warm_identity):
 folder=args.out/kind/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
 aux_weight=0. if kind=='full_no_aux' else .2
 signature=dict(pipeline=pc.read_json(args.out/'pipeline_signature.json'),kind=kind,branch_seed=seed,
  warmstart=warm_identity,epochs=args.epochs,batch=args.batch,audio_lr=args.audio_lr,fusion_lr=args.fusion_lr,
  weight_decay=.001,aux_weight=aux_weight,label_smoothing=.05,residual_l2=.001,dropout_av=.15,
  selected='global_validation_TAV_macro_f1_including_epoch_zero',permutations=args.permutations,test_evaluated=False)
 donefile=folder/'complete.json'
 if donefile.exists():
  old=pc.read_json(donefile)
  if old['status']!='complete' or old['signature']!=signature:raise RuntimeError('Completed fusion job configuration changed')
  for rel,digest in old['artifacts'].items():
   if pc.sha_file(folder/rel)!=digest:raise RuntimeError(f'Fusion artifact changed: {folder/rel}')
  print('PANNS_FUSION_CACHE',kind,seed,flush=True);return old['summary']
 x=assigned_features(data,kind,donors);y=data['train']['y'];yy=data['valid']['y']
 torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 model=SupervisedFusion(x['train'].shape[1],contexts['train']['ht'].shape[1]).to(args.device)
 model.audio.load_state_dict(warm_state,strict=True)
 identity=dict(parameters=sum(p.numel() for p in model.parameters()),audio_parameters=sum(p.numel() for p in model.audio.parameters()),
  initialization_sha256=fingerprint(model),input_dim=x['train'].shape[1],text_dim=contexts['train']['ht'].shape[1],
  warmstart_sha256=warm_identity['checkpoint_sha256'],parent_frozen=True)
 pc.record_json(args.out/'initialization'/f'seed_{seed}.json',identity);pc.dump_json(folder/'identity.json',identity)
 plan=training_plan(len(y),args.epochs,seed);planfile=args.out/'training_plans'/f'seed_{seed}.npz'
 if planfile.exists():
  prior=pc.load_npz(planfile)
  if set(prior)!=set(plan) or any(not np.array_equal(prior[k],v) for k,v in plan.items()):raise RuntimeError('Training plan differs')
 else:pc.atomic_npz(planfile,**plan)
 counts=np.bincount(y,minlength=2)
 if (counts==0).any():raise RuntimeError('Both training labels are required')
 w=np.sqrt(counts.sum()/(2*counts));w=w/w.mean()
 ce=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=args.device),label_smoothing=.05)
 optimizer=torch.optim.AdamW([dict(params=model.audio.parameters(),lr=args.audio_lr),
  dict(params=list(model.gate.parameters())+[model.strength],lr=args.fusion_lr)],weight_decay=.001)
 initial=predict(model,contexts['valid'],x['valid'],'TAV',args.device)[0]
 if not np.array_equal(initial,contexts['valid']['logits_TAV']):raise RuntimeError('Initialization is not exact parent identity')
 best=info.metrics(yy,initial)['macro_f1'];best_epoch=0;history=[];started=time.monotonic()
 save_weights(folder/'best.pt',state(model))
 print('PANNS_FUSION_START',kind,seed,'parameters',identity['parameters'],'baseline_f1',round(best,6),flush=True)
 # Index a+2*v corresponds to T,TA,TV,TAV. Both dropout masks are independent.
 bases=np.stack([contexts['train']['logits_'+m] for m in MODES],axis=1)
 for epoch in range(1,args.epochs+1):
  tick=time.monotonic();model.train();loss_sum=0.;aux_sum=0.;seen=0
  for start in range(0,len(y),args.batch):
   ix=plan['order'][epoch-1,start:start+args.batch]
   present=plan['audio_present'][epoch-1,ix];vision=plan['vision_present'][epoch-1,ix]
   mode_ix=present.astype(np.int64)+2*vision.astype(np.int64)
   base=torch.as_tensor(bases[ix,mode_ix],device=args.device)
   ht=torch.as_tensor(contexts['train']['ht'][ix],device=args.device)
   target=torch.as_tensor(y[ix],device=args.device).long();a_mask=torch.as_tensor(present,device=args.device)
   optimizer.zero_grad(set_to_none=True)
   logits,audio,delta,gate=model(base,ht,torch.as_tensor(x['train'][ix],device=args.device),a_mask)
   aux=ce(audio[a_mask],target[a_mask]) if present.any() else audio.sum()*0
   loss=ce(logits,target)+aux_weight*aux+.001*delta.square().mean()
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite fusion loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
   loss_sum+=float(loss.detach().cpu())*len(ix);aux_sum+=float(aux.detach().cpu())*len(ix);seen+=len(ix)
  logits,audio,delta,gate=predict(model,contexts['valid'],x['valid'],'TAV',args.device)
  value=info.metrics(yy,logits);audio_value=info.metrics(yy,audio)
  if value['macro_f1']>best:
   best=value['macro_f1'];best_epoch=epoch;save_weights(folder/'best.pt',state(model))
  row=dict(epoch=epoch,train_loss=loss_sum/seen,train_audio_aux_ce=aux_sum/seen,val_macro_f1=value['macro_f1'],
    val_audio_macro_f1=audio_value['macro_f1'],val_accuracy=value['accuracy'],strength=float(torch.tanh(model.strength).detach().cpu()),
    delta_mean_abs=float(np.abs(delta).mean()),gate_mean=float(gate.mean()),seconds=time.monotonic()-tick)
  history.append(row);pc.dump_json(folder/'history.json',history)
  pc.dump_json(args.out/'current_job.json',dict(kind=kind,seed=seed,epoch=epoch,epochs=args.epochs,best_val_f1=best,best_epoch=best_epoch))
  print('PANNS_FUSION_EPOCH',kind,seed,epoch,'val_f1',round(value['macro_f1'],6),'audio_f1',round(audio_value['macro_f1'],6),
        'seconds',round(row['seconds'],2),flush=True)
 model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True));model.eval()
 arrays=dict(ids=data['valid']['ids'],labels=yy,assigned_donors=donors['valid'] if kind=='shuffled_supervised' else np.arange(len(yy)))
 rows=[];mode_metrics={}
 def add(tag,condition,mode,rep,logits,delta,gate,reference,ix=None):
  p=logits.argmax(1);q=info.prob(logits);r=reference.argmax(1);rq=info.prob(reference)
  row=dict(condition=condition,mode=mode,rep=rep,**info.metrics(yy,logits),prediction_changes=int((p!=r).sum()),
   corrected=int(((r!=yy)&(p==yy)).sum()),broken=int(((r==yy)&(p!=yy)).sum()),
   mean_abs_probability_change=float(np.abs(q-rq).mean()),mean_abs_delta=float(np.abs(delta).mean()),gate_mean=float(gate.mean()))
  rows.append(row);arrays['pred_'+tag]=p;arrays['prob_'+tag]=q;arrays['delta_'+tag]=delta;arrays['gate_'+tag]=gate
  if ix is not None:arrays['donor_'+tag]=ix
  return row
 assigned_logits=None
 for mode in MODES:
  logits,audio,delta,gate=predict(model,contexts['valid'],x['valid'],mode,args.device)
  if mode in ('T','TV') and not np.array_equal(logits,contexts['valid']['logits_'+mode]):raise RuntimeError('Missing audio changed parent logits')
  row=add(mode,'assigned',mode,-1,logits,delta,gate,contexts['valid']['logits_'+mode]);mode_metrics[mode]=info.metrics(yy,logits)
  if mode=='TAV':
   assigned_logits=logits;arrays['audio_pred']=audio.argmax(1);arrays['audio_prob']=info.prob(audio)
 if abs(mode_metrics['TAV']['macro_f1']-best)>1e-12:raise RuntimeError('Best fusion checkpoint replay differs')
 logits,audio,delta,gate=predict(model,contexts['valid'],np.zeros_like(x['valid']),'TAV',args.device)
 add('zero','zero_input','TAV',-1,logits,delta,gate,assigned_logits)
 for rep in range(args.permutations):
  ix=info.cross_video(data['valid']['ids'],20263028+rep)
  features=np.zeros_like(x['valid']) if kind=='zero_supervised' else data['valid']['x'][ix]
  logits,audio,delta,gate=predict(model,contexts['valid'],features,'TAV',args.device)
  add('cross_video_'+str(rep).zfill(2),'cross_video','TAV',rep,logits,delta,gate,assigned_logits,ix)
 pc.atomic_npz(folder/'validation_predictions.npz',**arrays);pc.write_csv(folder/'validation_metrics.csv',rows)
 perturbed=[r['macro_f1'] for r in rows if r['condition']=='cross_video'];parent=info.metrics(yy,contexts['valid']['logits_TAV'])
 summary=dict(kind=kind,seed=seed,best_epoch=best_epoch,identity=identity,modes=mode_metrics,parent=parent,
  gain_over_parent_pp=100*(best-parent['macro_f1']),permuted_macro_f1_mean=float(np.mean(perturbed)),
  assigned_minus_permuted_pp=100*(best-float(np.mean(perturbed))),
  correspondence_interpretation='true_alignment_vs_cross_video' if kind in ('full_supervised','full_no_aux') else 'negative_control_assignment_vs_other_assignments',
  corrected=next(r['corrected'] for r in rows if r['condition']=='assigned' and r['mode']=='TAV'),
  broken=next(r['broken'] for r in rows if r['condition']=='assigned' and r['mode']=='TAV'),
  missing_audio_exact_identity=True,selected_strength=float(torch.tanh(model.strength).detach().cpu()),
  elapsed_seconds=time.monotonic()-started,test_evaluated=False)
 files=('best.pt','identity.json','history.json','validation_predictions.npz','validation_metrics.csv')
 pc.dump_json(donefile,dict(status='complete',signature=signature,summary=summary,artifacts={f:pc.sha_file(folder/f) for f in files},test_evaluated=False))
 print('PANNS_FUSION_DONE',kind,seed,'best_epoch',best_epoch,'gain_pp',round(summary['gain_over_parent_pp'],4),flush=True)
 del model;gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()
 return summary

def verify_warmstart(data,folder,warm_state,device):
 model=info.Probe('panns_embedding',data['x'].shape[1]).to(device)
 model.load_state_dict(warm_state,strict=True)
 logits=info.predict(model,data['x'],None,device);previous=pc.load_npz(folder/'validation_predictions.npz')
 if not np.array_equal(data['ids'],previous['ids']) or not np.array_equal(data['y'],previous['labels']):raise RuntimeError('Warmstart validation IDs differ')
 if not np.array_equal(logits.argmax(1),previous['pred_original']) or np.max(np.abs(info.prob(logits)-previous['prob_original']))>1e-5:
  raise RuntimeError('Audio warmstart readout replay differs')
 return dict(max_probability_error=float(np.max(np.abs(info.prob(logits)-previous['prob_original']))),classes_equal=True)

def summarize(args,jobs):
 rows=[]
 for kind in KINDS:
  selected=[j for j in jobs if j['kind']==kind]
  if len(selected)!=len(args.seeds):raise RuntimeError('Missing branch seed')
  rows.append(dict(kind=kind,seeds=len(selected),parameters=selected[0]['identity']['parameters'],
   macro_f1_mean=float(np.mean([j['modes']['TAV']['macro_f1'] for j in selected])),
   macro_f1_sd=float(np.std([j['modes']['TAV']['macro_f1'] for j in selected],ddof=1)) if len(selected)>1 else 0.,
   gain_over_parent_pp=float(np.mean([j['gain_over_parent_pp'] for j in selected])),
   assigned_minus_permuted_pp=float(np.mean([j['assigned_minus_permuted_pp'] for j in selected])),
   corrected_mean=float(np.mean([j['corrected'] for j in selected])),broken_mean=float(np.mean([j['broken'] for j in selected]))))
 pc.write_csv(args.out/'fusion_summary.csv',rows)
 pc.dump_json(args.out/'summary.json',dict(status='complete',scope='frozen_paper_TRIM_plus_supervised_full_PANNs_residual',
  baseline_parent_seed=42,branch_seeds=args.seeds,results=jobs,summary=rows,test_evaluated=False,
  caveats=['Validation-selected exploratory continuation after previous validation probes; not independent confirmation',
   'Branch seeds share one fixed TRIM checkpoint, not three independent backbone runs',
   'Full no-aux still starts from a supervised audio classifier',
   'PANNs embedding mixes speech, prosody and background sound; this is not isolated environmental-sound evidence',
   'External pretraining and additional parameters must be reported',
   'Current media availability subset is biased and cannot substitute for full-dataset test evidence']))
 # Export only small branch artifacts and validation features, never parent weights or training embeddings.
 files=[p for p in args.out.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.npz','.pt')
  and p.name!='review_manifest.json' and 'cache' not in p.relative_to(args.out).parts
  and 'training_plans' not in p.relative_to(args.out).parts]
 manifest={p.relative_to(args.out).as_posix():pc.sha_file(p) for p in files}
 for name,source in (('PROTOCOL.md',HERE/'PROTOCOL.md'),('source_manifest.json',HERE/'bundle_manifest.json'),
                     ('fusion_core.py',HERE/'fusion_core.py'),('run_fusion.py',HERE/'run_fusion.py')):
  manifest[name]=pc.sha_file(source)
 pc.dump_json(args.out/'review_manifest.json',dict(files=manifest,test_evaluated=False))
 archive=args.out/'panns_fusion_review_20261028.zip';temp=archive.with_suffix('.tmp.zip')
 with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
  for p in files+[args.out/'review_manifest.json']:z.write(p,'panns_fusion_review/'+p.relative_to(args.out).as_posix())
  for name in ('PROTOCOL.md','fusion_core.py','run_fusion.py'):z.write(HERE/name,'panns_fusion_review/'+name)
  z.write(HERE/'bundle_manifest.json','panns_fusion_review/source_manifest.json')
 os.replace(temp,archive);print('PANNS_FUSION_REVIEW_COMPLETE',archive,flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=HERE.parent.parent)
 ap.add_argument('--parent-run',type=Path,default=PARENT/'runs/pilot');ap.add_argument('--causal-run',type=Path,default=CAUSAL/'runs/pilot')
 ap.add_argument('--info-run',type=Path,default=INFO/'runs/pilot');ap.add_argument('--out',type=Path,default=HERE/'runs/pilot')
 ap.add_argument('--device',default='cuda:0');ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--batch',type=int,default=64)
 ap.add_argument('--audio-lr',type=float,default=.0001);ap.add_argument('--fusion-lr',type=float,default=.001)
 ap.add_argument('--permutations',type=int,default=20);ap.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS));args=ap.parse_args()
 if min(args.epochs,args.batch,args.permutations,args.audio_lr,args.fusion_lr)<=0 or len(set(args.seeds))!=len(args.seeds):raise ValueError('Invalid training configuration')
 if any(seed not in SEEDS for seed in args.seeds):raise ValueError('No matching supervised warmstart for requested seed')
 release=verify_release();args.out.mkdir(parents=True,exist_ok=True);args.device=info.runtime(args.device)
 data,contexts,data_signature,donors=prepare(args)
 signature=dict(release=release,data=data_signature,seeds=args.seeds,epochs=args.epochs,batch=args.batch,permutations=args.permutations,
  audio_lr=args.audio_lr,fusion_lr=args.fusion_lr,torch=torch.__version__,numpy=np.__version__,device_type=args.device.type,test_evaluated=False)
 pc.record_json(args.out/'pipeline_signature.json',signature)
 jobs=[]
 for seed in args.seeds:
  folder,done=audio_source(args.info_run,seed)
  if done['signature']['pipeline']!=pc.read_json(args.info_run/'pipeline_signature.json'):raise RuntimeError('Warmstart training pipeline differs')
  weights=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
  replay=verify_warmstart(data['valid'],folder,weights,args.device)
  warm_identity=dict(checkpoint_sha256=pc.sha_file(folder/'best.pt'),complete_sha256=pc.sha_file(folder/'complete.json'),seed=seed,
   selected_audio_epoch=done['summary']['best_epoch'],replay=replay)
  for kind in KINDS:
   jobs.append(fit(args,data,contexts,kind,seed,donors,weights,warm_identity))
   pc.dump_json(args.out/'suite_progress.json',dict(done=len(jobs),total=len(KINDS)*len(args.seeds),status='running',test_evaluated=False))
 pc.dump_json(args.out/'suite_progress.json',dict(done=len(jobs),total=len(jobs),status='complete',test_evaluated=False))
 summarize(args,jobs);print('PANNS_FUSION_SUITE_COMPLETE test_evaluated=False',flush=True)
if __name__=='__main__':main()