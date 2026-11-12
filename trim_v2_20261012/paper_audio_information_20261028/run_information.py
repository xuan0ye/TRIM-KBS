"""Remote small-model diagnostics of audio information and aligned-feature order."""
import argparse,gc,time,zipfile
from information_core import *

def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def save_weights(path,weights):
 temp=path.with_suffix('.tmp');torch.save(weights,temp);os.replace(temp,path)
def fit(args,data,recipe,seed,label_donors):
 folder=args.out/recipe/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
 signature=dict(pipeline=pc.read_json(args.out/'pipeline_signature.json'),recipe=recipe,seed=seed,
   train_labels='fixed_permuted_train_only' if recipe=='label_shuffled_cnn' else 'original',
   validation_labels='original',selected='validation_macro_f1',epochs=args.epochs,lr=.001,weight_decay=.001,
   batch=args.batch,label_smoothing=.05,fp32=True,test_evaluated=False)
 donefile=folder/'complete.json'
 if donefile.exists():
  old=pc.read_json(donefile)
  if old['signature']!=signature:raise RuntimeError(f'Completed job configuration changed: {folder}')
  for rel,digest in old['artifacts'].items():
   if pc.sha_file(folder/rel)!=digest:raise RuntimeError(f'Artifact changed: {folder/rel}')
  print('AUDIO_PROBE_CACHE',recipe,seed,flush=True);return old['summary']
 xtr,mtr=representation(data['train'],recipe);xva,mva=representation(data['valid'],recipe)
 y=data['train']['y'].copy()
 if recipe=='label_shuffled_cnn':y=y[label_donors]
 torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 model=Probe(recipe,xtr.shape[-1]).to(args.device)
 fingerprint=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
 identity=dict(parameter_count=sum(p.numel() for p in model.parameters()),initialization_sha256=fingerprint,
               architecture='temporal_cnn' if recipe in CNN_RECIPES else 'mlp32',input_dim=xtr.shape[-1])
 family='cnn' if recipe in CNN_RECIPES else 'mlp_'+str(xtr.shape[-1])
 pc.record_json(args.out/'initialization'/f'{family}_seed_{seed}.json',identity)
 pc.dump_json(folder/'identity.json',identity)
 counts=np.bincount(y,minlength=2);w=np.sqrt(counts.sum()/(2*counts));w=w/w.mean()
 ce=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=args.device),label_smoothing=.05)
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
 rng=np.random.default_rng(seed);best=-1.;best_epoch=0;history=[];started=time.monotonic()
 print('AUDIO_PROBE_START',recipe,seed,'train',len(y),'valid',len(data['valid']['y']),flush=True)
 for epoch in range(1,args.epochs+1):
  tick=time.monotonic();model.train();order=rng.permutation(len(y));loss_sum=0.;count=0
  for start in range(0,len(y),args.batch):
   ix=order[start:start+args.batch];optimizer.zero_grad(set_to_none=True)
   output=model(torch.as_tensor(xtr[ix],device=args.device),None if mtr is None else torch.as_tensor(mtr[ix],device=args.device))
   loss=ce(output,torch.as_tensor(y[ix],device=args.device).long())
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite probe loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
   loss_sum+=float(loss.detach().cpu())*len(ix);count+=len(ix)
  logits=predict(model,xva,mva,args.device);value=metrics(data['valid']['y'],logits)
  if value['macro_f1']>best:
   best=value['macro_f1'];best_epoch=epoch;save_weights(folder/'best.pt',state(model))
  row=dict(epoch=epoch,train_loss=loss_sum/count,val_macro_f1=value['macro_f1'],val_accuracy=value['accuracy'],seconds=time.monotonic()-tick)
  history.append(row);pc.dump_json(folder/'history.json',history)
  pc.dump_json(args.out/'current_job.json',dict(recipe=recipe,seed=seed,epoch=epoch,epochs=args.epochs,best_val_f1=best))
  print('AUDIO_PROBE_EPOCH',recipe,seed,epoch,'val_f1',round(value['macro_f1'],6),'seconds',round(row['seconds'],2),flush=True)
 model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True));model.eval()
 clean=predict(model,xva,mva,args.device);base=metrics(data['valid']['y'],clean)
 if abs(base['macro_f1']-best)>1e-12:raise RuntimeError('Selected checkpoint replay differs')
 arrays=dict(ids=data['valid']['ids'],labels=data['valid']['y'],pred_original=clean.argmax(1),prob_original=prob(clean))
 rows=[dict(condition='original',rep=-1,**base,prediction_changes=0,corrected=0,broken=0,mean_abs_probability_change=0.)]
 original_pred=clean.argmax(1);original_prob=prob(clean);yy=data['valid']['y']
 def add(tag,kind,rep,logits,donors=None):
  p=logits.argmax(1);q=prob(logits);m=metrics(yy,logits)
  rows.append(dict(condition=kind,rep=rep,**m,prediction_changes=int((p!=original_pred).sum()),
   corrected=int(((original_pred!=yy)&(p==yy)).sum()),broken=int(((original_pred==yy)&(p!=yy)).sum()),
   mean_abs_probability_change=float(np.abs(q-original_prob).mean())))
  arrays['pred_'+tag]=p;arrays['prob_'+tag]=q
  if donors is not None:arrays['donor_'+tag]=donors
 for rep in range(args.permutations):
  ix=cross_video(data['valid']['ids'],20261028+rep)
  logits=predict(model,xva[ix],None if mva is None else mva[ix],args.device)
  add('cross_video_'+str(rep).zfill(2),'cross_video',rep,logits,ix)
 if recipe in CNN_RECIPES:
  for rep in range(args.permutations):
   shuffled=frame_permute(xva,mva,data['valid']['ids'],20262028+rep)
   logits=predict(model,shuffled,mva,args.device)
   add('frame_shuffle_'+str(rep).zfill(2),'frame_shuffle',rep,logits)
  reversed_x=xva.copy()
  for i in range(len(xva)):
   positions=np.flatnonzero(mva[i]);reversed_x[i,positions]=xva[i,positions[::-1]]
  add('reversed','reversed',-1,predict(model,reversed_x,mva,args.device))
 pc.atomic_npz(folder/'validation_predictions.npz',**arrays)
 pc.write_csv(folder/'validation_metrics.csv',rows)
 controls=[]
 for kind in ('cross_video','frame_shuffle'):
  chosen=[r for r in rows if r['condition']==kind]
  if chosen:
   controls.append(dict(intervention=kind,repetitions=len(chosen),permuted_macro_f1_mean=float(np.mean([r['macro_f1'] for r in chosen])),
    correspondence_gain_pp=100*(base['macro_f1']-float(np.mean([r['macro_f1'] for r in chosen]))),
    mean_prediction_changes=float(np.mean([r['prediction_changes'] for r in chosen]))))
 summary=dict(recipe=recipe,seed=seed,best_epoch=best_epoch,parameter_count=identity['parameter_count'],metrics=base,controls=controls,
  order_scope='aligned_audio_feature_order_only' if recipe in CNN_RECIPES else 'not_applicable_to_precomputed_PANNs' if recipe=='panns_embedding' else 'permutation_invariant_summary_by_definition',
  elapsed_seconds=time.monotonic()-started,test_evaluated=False)
 artifacts={f:pc.sha_file(folder/f) for f in ('best.pt','identity.json','history.json','validation_predictions.npz','validation_metrics.csv')}
 pc.dump_json(donefile,dict(status='complete',signature=signature,summary=summary,artifacts=artifacts,test_evaluated=False))
 print('AUDIO_PROBE_COMPLETE',json.dumps(summary),flush=True)
 del model;gc.collect()
 if torch.cuda.is_available():torch.cuda.empty_cache()
 return summary

def summarize(args,data,jobs):
 rows=[]
 for recipe in RECIPES:
  chosen=[j for j in jobs if j['recipe']==recipe]
  if len(chosen)!=len(args.seeds):raise RuntimeError('Missing probe seed')
  rows.append(dict(recipe=recipe,seeds=len(chosen),parameters=chosen[0]['parameter_count'],
   macro_f1_mean=float(np.mean([j['metrics']['macro_f1'] for j in chosen])),
   macro_f1_sd=float(np.std([j['metrics']['macro_f1'] for j in chosen],ddof=1)) if len(chosen)>1 else 0.,
   balanced_accuracy_mean=float(np.mean([j['metrics']['balanced_accuracy'] for j in chosen])),
   accuracy_mean=float(np.mean([j['metrics']['accuracy'] for j in chosen]))))
 pc.write_csv(args.out/'probe_summary.csv',rows)
 y=data['valid']['y'];prior=float(data['train']['y'].mean());p=np.full(len(y),int(prior>.5),dtype=np.int8)
 dummy=pc.scores(y,p,np.full(len(y),prior,dtype=np.float32))
 summary=dict(status='complete',scope='audio_only_information_probe_not_new_trim_score',train_n=len(data['train']['y']),valid_n=len(y),
  seeds=args.seeds,recipes=list(RECIPES),summary=rows,training_prior_dummy=dummy,results=jobs,test_evaluated=False,
  caveats=['Selected validation estimates, not test results','Nonzero rows are an observed-mask proxy, not known physical durations',
    'Length reflects aligned representation and can proxy linguistic length','PANNs uses external pretraining and larger input dimension',
    'Bag baselines have different parameter counts; temporal/within-frame-shuffled/label-shuffled CNNs share architecture and initialization',
    'Aligned feature order is not raw waveform segment order','Positive audio-only prediction does not guarantee complementary TRIM gains'])
 pc.dump_json(args.out/'summary.json',summary)
 files=[p for p in args.out.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.npz','.pt')
       and p.name not in ('dataset_train.npz','dataset_valid.npz','review_manifest.json')]
 manifest={p.relative_to(args.out).as_posix():pc.sha_file(p) for p in files}
 manifest['PROTOCOL.md']=pc.sha_file(HERE/'PROTOCOL.md');manifest['source_manifest.json']=pc.sha_file(HERE/'bundle_manifest.json')
 pc.dump_json(args.out/'review_manifest.json',dict(files=manifest,test_evaluated=False))
 archive=args.out/'audio_information_review_20261028.zip';tmp=archive.with_suffix('.tmp.zip')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
  for p in files+[args.out/'review_manifest.json']:z.write(p,'audio_information_review/'+p.relative_to(args.out).as_posix())
  z.write(HERE/'PROTOCOL.md','audio_information_review/PROTOCOL.md')
  z.write(HERE/'bundle_manifest.json','audio_information_review/source_manifest.json')
 os.replace(tmp,archive)
 print('AUDIO_INFORMATION_REVIEW_COMPLETE',archive,flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=HERE.parent.parent)
 ap.add_argument('--run',type=Path,default=PARENT/'runs/pilot');ap.add_argument('--out',type=Path,default=HERE/'runs/pilot')
 ap.add_argument('--device',default='cuda:0');ap.add_argument('--epochs',type=int,default=15)
 ap.add_argument('--batch',type=int,default=64);ap.add_argument('--permutations',type=int,default=20)
 ap.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS));args=ap.parse_args()
 if min(args.epochs,args.batch,args.permutations)<1 or len(set(args.seeds))!=len(args.seeds):raise ValueError('Invalid configuration')
 release=verify_release();args.out.mkdir(parents=True,exist_ok=True);args.device=runtime(args.device)
 data,label_donors,data_signature=prepare(args.root,args.run,args.out)
 signature=dict(release=release,data=data_signature,seeds=args.seeds,epochs=args.epochs,batch=args.batch,
  permutations=args.permutations,torch=torch.__version__,numpy=np.__version__,device_type=args.device.type,
  data_files={f:pc.sha_file(args.out/f) for f in ('dataset_train.npz','dataset_valid.npz','normalization.npz','data_index.npz')},
  test_evaluated=False)
 pc.record_json(args.out/'pipeline_signature.json',signature)
 jobs=[]
 for recipe in RECIPES:
  for seed in args.seeds:
   jobs.append(fit(args,data,recipe,seed,label_donors))
   pc.dump_json(args.out/'suite_progress.json',dict(done=len(jobs),total=len(RECIPES)*len(args.seeds),status='running',test_evaluated=False))
 pc.dump_json(args.out/'suite_progress.json',dict(done=len(jobs),total=len(jobs),status='complete',test_evaluated=False))
 summarize(args,data,jobs);print('AUDIO_INFORMATION_SUITE_COMPLETE test_evaluated=False',flush=True)
if __name__=='__main__':main()