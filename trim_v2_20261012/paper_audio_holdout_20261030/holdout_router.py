"""Small probability routers trained only on expert-unseen source videos."""
from holdout_core import *

class ProbabilityRouter(nn.Module):
 def __init__(self,constant=False):
  super().__init__();self.constant=constant;self.register_buffer('enabled',torch.tensor(0.))
  if constant:self.bias=nn.Parameter(torch.sigmoid(torch.tensor(-2.)))
  else:
   self.net=nn.Sequential(nn.Linear(7,16),nn.GELU(),nn.Linear(16,1))
   nn.init.zeros_(self.net[-1].weight);nn.init.constant_(self.net[-1].bias,-2.)
 def activate(self):self.enabled.fill_(1.)
 def project(self):
  if self.constant:
   with torch.no_grad():self.bias.clamp_(0,1)
 def forward(self,parent_logits,audio_logits,audio_present):
  base=parent_logits.detach();expert=audio_logits.detach();p=base.softmax(-1);q=expert.softmax(-1)
  if self.constant:gate=self.bias.clamp(0,1).expand(len(base))
  else:
   pp=p[:,1];qq=q[:,1]
   entropy=lambda t:-(t.clamp_min(1e-30)*t.clamp_min(1e-30).log()).sum(1)/np.log(2.)
   features=torch.stack((pp,qq,entropy(p),entropy(q),(pp-qq).abs(),(2*pp-1).abs(),(2*qq-1).abs()),dim=-1)
   gate=torch.sigmoid(self.net(features).squeeze(-1))
  weight=gate*self.enabled*audio_present.to(gate.dtype)
  mixed=p+weight[:,None]*(q-p)
  # Subtract the shift before adding it so disabled baseline and missing audio are bit-identical.
  output=base+(mixed.clamp_min(1e-30).log()-p.clamp_min(1e-30).log())
  return output,weight

def routed_audio(readout,kind,donor):
 if kind=='gated_zero':return readout['zero_audio_logits']
 if kind=='gated_shuffled':return readout['audio_logits'][donor]
 return readout['audio_logits']

def predict_router(model,readout,audio,mode,device,batch=256):
 model.eval();outputs=[];weights=[]
 with torch.inference_mode():
  for start in range(0,len(readout['y']),batch):
   sl=slice(start,start+batch);n=len(readout['y'][sl])
   logits,w=model(torch.as_tensor(readout['logits_'+mode][sl],device=device),torch.as_tensor(audio[sl],device=device),
    torch.full((n,),pc.MASKS[mode][1],device=device))
   outputs.append(logits.cpu().numpy());weights.append(w.cpu().numpy())
 return np.concatenate(outputs),np.concatenate(weights)

def train_router(args,kind,train,dev,pipeline):
 folder=args.out/'routers'/kind;folder.mkdir(parents=True,exist_ok=True)
 signature=dict(pipeline=pipeline,kind=kind,seed=args.seed,epochs=args.router_epochs,batch=64,lr=.001,weight_decay=.001,
  gate_penalty=.001,modality_dropout=.15,fit_role='router_train',selection_role='router_dev',
  selection='strict_macro_f1_including_disabled_original_parent',test_evaluated=False)
 completed=check_completed(folder,signature)
 if completed is not None:return completed
 donors={s:cross_video(z['ids'],20261030+i) for i,(s,z) in enumerate((('train',train),('dev',dev)))}
 a_train=routed_audio(train,kind,donors['train']);a_dev=routed_audio(dev,kind,donors['dev'])
 seed_all(args.seed);model=ProbabilityRouter(kind=='constant_correct').to(args.device)
 initial=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
 identity=dict(parameters=sum(p.numel() for p in model.parameters()),initialization_sha256=initial)
 family='constant' if kind=='constant_correct' else 'gated'
 pc.record_json(args.out/'router_initialization'/(family+'.json'),identity)
 initial_logits,_=predict_router(model,dev,a_dev,'TAV',args.device)
 if not np.array_equal(initial_logits,dev['logits_TAV']):raise RuntimeError('Disabled router must exactly reproduce parent')
 best=scores(dev['y'],initial_logits)['macro_f1'];best_epoch=0;history=[]
 pt.save_torch(folder/'best.pt',pt.cpu_state(model));model.activate()
 optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);ce=criterion(train['y'],args.device)
 rng=np.random.default_rng(args.seed);base=np.stack([train['logits_'+m] for m in MODES],1)
 print('HOLDOUT_ROUTER_START',kind,'params',identity['parameters'],'router_dev_baseline_f1',round(best,6),flush=True)
 for epoch in range(1,args.router_epochs+1):
  model.train();order=rng.permutation(len(train['y']));present_a=rng.random(len(order))>=.15;present_v=rng.random(len(order))>=.15;loss_total=0.
  for start in range(0,len(order),64):
   ix=order[start:start+64];a=present_a[ix];v=present_v[ix];mode=a.astype(np.int64)+2*v.astype(np.int64)
   optimizer.zero_grad(set_to_none=True)
   logits,w=model(torch.as_tensor(base[ix,mode],device=args.device),torch.as_tensor(a_train[ix],device=args.device),torch.as_tensor(a,device=args.device))
   loss=ce(logits,torch.as_tensor(train['y'][ix],device=args.device).long())+.001*w.square().mean()
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite router loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();model.project();loss_total+=float(loss.detach().cpu())*len(ix)
  logits,w=predict_router(model,dev,a_dev,'TAV',args.device);m=scores(dev['y'],logits)
  if m['macro_f1']>best:best=m['macro_f1'];best_epoch=epoch;pt.save_torch(folder/'best.pt',pt.cpu_state(model))
  history.append(dict(epoch=epoch,train_loss=loss_total/len(train['y']),router_dev_macro_f1=m['macro_f1'],router_dev_nll=m['nll'],mean_audio_weight=float(w.mean())))
  pc.dump_json(folder/'history.json',history)
  pc.dump_json(args.out/'current_job.json',dict(stage='router',kind=kind,epoch=epoch,epochs=args.router_epochs,best_router_dev_f1=best))
  if epoch==1 or epoch%5==0:print('HOLDOUT_ROUTER_EPOCH',kind,epoch,'router_dev_f1',round(m['macro_f1'],6),'weight',round(float(w.mean()),4),flush=True)
 model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True),strict=True)
 logits,w=predict_router(model,dev,a_dev,'TAV',args.device)
 if abs(scores(dev['y'],logits)['macro_f1']-best)>1e-12:raise RuntimeError('Router internal best replay differs')
 pc.atomic_npz(folder/'router_dev_predictions.npz',ids=dev['ids'],labels=dev['y'],pred=logits.argmax(1),prob=prob(logits),weight=w,donors=donors['dev'] if kind=='gated_shuffled' else np.arange(len(dev['y'])))
 done=dict(status='complete',signature=signature,identity=identity,best_epoch=best_epoch,best_router_dev_macro_f1=best,
  branch_enabled=bool(float(model.enabled)),artifacts={f:pc.sha_file(folder/f) for f in ('best.pt','history.json','router_dev_predictions.npz')},test_evaluated=False)
 pc.dump_json(folder/'complete.json',done);return done

def external_evaluation(args,external,jobs,seal):
 if pc.read_json(args.out/'selection_seal.json')!=seal:raise RuntimeError('External evaluation requires sealed selections')
 base={m:scores(external['y'],external['logits_'+m]) for m in MODES}
 donors=cross_video(external['ids'],20261032);all_rows=[];results=[]
 for kind in CONDITIONS:
  folder=args.out/'routers'/kind
  if pc.sha_file(folder/'best.pt')!=seal['routers'][kind]:raise RuntimeError('Router changed after selection seal')
  model=ProbabilityRouter(kind=='constant_correct').to(args.device)
  model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True),strict=True)
  assigned=routed_audio(external,kind,donors);arrays=dict(ids=external['ids'],labels=external['y'],assigned_donors=donors if kind=='gated_shuffled' else np.arange(len(external['y'])))
  chosen_logits=None;metrics={}
  for mode in MODES:
   logits,w=predict_router(model,external,assigned,mode,args.device)
   if mode in ('T','TV') and not np.array_equal(logits,external['logits_'+mode]):raise RuntimeError('Missing audio changed paper parent')
   arrays['pred_'+mode]=logits.argmax(1);arrays['prob_'+mode]=prob(logits);arrays['weight_'+mode]=w
   m=scores(external['y'],logits);metrics[mode]=m
   parent=external['logits_'+mode].argmax(1);p=logits.argmax(1);y=external['y']
   all_rows.append(dict(kind=kind,condition='assigned',mode=mode,rep=-1,**m,corrected=int(((parent!=y)&(p==y)).sum()),broken=int(((parent==y)&(p!=y)).sum()),
    mean_audio_weight=float(w.mean())))
   if mode=='TAV':chosen_logits=logits
  clean=chosen_logits.argmax(1);permuted=[]
  zero_logits,zero_w=predict_router(model,external,external['zero_audio_logits'],'TAV',args.device)
  p=zero_logits.argmax(1);y=external['y'];m=scores(y,zero_logits)
  arrays['pred_zero']=p;arrays['prob_zero']=prob(zero_logits);arrays['weight_zero']=zero_w
  all_rows.append(dict(kind=kind,condition='zero_input',mode='TAV',rep=-1,**m,corrected=int(((clean!=y)&(p==y)).sum()),broken=int(((clean==y)&(p!=y)).sum()),mean_audio_weight=float(zero_w.mean())))
  for rep in range(args.permutations):
   ix=cross_video(external['ids'],20261130+rep)
   audio=external['zero_audio_logits'] if kind=='gated_zero' else external['audio_logits'][ix]
   logits,w=predict_router(model,external,audio,'TAV',args.device);m=scores(external['y'],logits);p=logits.argmax(1);y=external['y']
   tag='cross_video_'+str(rep).zfill(2);arrays['pred_'+tag]=p;arrays['prob_'+tag]=prob(logits);arrays['weight_'+tag]=w;arrays['donor_'+tag]=ix
   all_rows.append(dict(kind=kind,condition='cross_video',mode='TAV',rep=rep,**m,corrected=int(((clean!=y)&(p==y)).sum()),broken=int(((clean==y)&(p!=y)).sum()),mean_audio_weight=float(w.mean())))
   permuted.append(m['macro_f1'])
  pc.atomic_npz(folder/'external_predictions.npz',**arrays)
  results.append(dict(kind=kind,parameters=jobs[kind]['identity']['parameters'],best_router_dev_epoch=jobs[kind]['best_epoch'],
   internal_router_dev_f1=jobs[kind]['best_router_dev_macro_f1'],external_metrics=metrics,gain_over_same_expert_TAV_pp=100*(metrics['TAV']['macro_f1']-base['TAV']['macro_f1']),
   assigned_minus_permuted_pp=100*(metrics['TAV']['macro_f1']-float(np.mean(permuted)))))
 pc.write_csv(args.out/'external_metrics.csv',all_rows)
 return dict(baseline=base,audio_only=scores(external['y'],external['audio_logits']),results=results,
  external_role='original_reused_validation_not_independent_test',selection_sealed_before_external=True,test_evaluated=False)