"""Exact archived TRIM backbone/objective; validation-only representation controls."""
import argparse,gc,math,random,time
from unittest.mock import patch
from paper_common import *
import torch,transformers
from torch.utils.data import DataLoader,TensorDataset
from transformers import BertModel,BertConfig,get_cosine_schedule_with_warmup
from model_tools import runtime,encode,fuse
import model_legacy as legacy

def make_model(bert,a_dim,v_dim,pretrained=True):
    original=BertModel.from_pretrained
    def load(*args,**kwargs):
        if pretrained:
            model,info=original(str(bert),local_files_only=True,attn_implementation='sdpa',output_loading_info=True)
            if info['missing_keys'] or info['mismatched_keys'] or info['error_msgs']:
                raise RuntimeError(f'BERT did not load fully: {info}')
            return model
        config=BertConfig.from_pretrained(str(bert),local_files_only=True);config._attn_implementation='sdpa'
        return BertModel(config)
    with patch.object(legacy.BertModel,'from_pretrained',side_effect=load):
        return legacy.E2EMSAModel(str(bert),a_dim,v_dim,d_model=256,n_classes=2)

def save_torch(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)

def cpu_state(model): return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def initial_digest(model):
    # BERT tensors are independently verified; hash all randomly initialized paper parameters.
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        if not k.startswith('bert.'):
            h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)

def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda'] is not None: torch.cuda.set_rng_state_all(r['cuda'])

def loader(data,batch,shuffle=False,generator=None):
    return DataLoader(TensorDataset(*(torch.as_tensor(data[k]) for k in ('text','audio','vision','y'))),
                      batch_size=batch,shuffle=shuffle,num_workers=0,generator=generator)

def evaluate(model,data,batch,device,all_masks=False):
    model.eval();output={};predictions={};conditions=MASKS if all_masks else {'TAV':MASKS['TAV']};parity=0.;invariance=0.
    with torch.inference_mode():
        for tb,ta,tv,y in loader(data,batch):
            tb,ta,tv=tb.to(device),ta.to(device),tv.to(device)
            h=encode(model,tb,ta,tv)
            for key,bits in conditions.items():
                mask=torch.tensor(bits,device=device).expand(len(tb),-1)
                logits=fuse(model,*h,mask)
                output.setdefault(key,[]).append(logits.softmax(-1)[:,1].cpu().numpy())
                predictions.setdefault(key,[]).append(logits.argmax(-1).cpu().numpy())
                if all_masks and key=='TAV':
                    direct,_=model(tb[:,0],tb[:,1],tb[:,2],ta,tv,training=False)
                    parity=max(parity,float((direct-logits).abs().max()))
                if all_masks and key in ('T','TV'):
                    # Absent audio must not depend on its stored feature vector.
                    changed=fuse(model,h[0],torch.randn_like(h[1])*100,h[2],mask)
                    invariance=max(invariance,float((changed-logits).abs().max()))
            if all_masks:
                zero=encode(model,tb,torch.zeros_like(ta),tv)
                logits=fuse(model,*zero,torch.ones((len(tb),3),device=device))
                output.setdefault('TAV_zero_audio_present',[]).append(logits.softmax(-1)[:,1].cpu().numpy())
                predictions.setdefault('TAV_zero_audio_present',[]).append(logits.argmax(-1).cpu().numpy())
    if max(parity,invariance)>1e-5: raise RuntimeError(f'Forward/missing-slot parity failed: {parity}, {invariance}')
    output={k:np.concatenate(v) for k,v in output.items()}
    predictions={k:np.concatenate(v) for k,v in predictions.items()}
    metrics={k:scores(data['y'],predictions[k],p) for k,p in output.items()}
    arrays={'prob_'+k:v for k,v in output.items()};arrays.update({'pred_'+k:v for k,v in predictions.items()})
    return arrays,metrics,dict(forward_parity_max_delta=parity,absent_audio_max_delta=invariance)

def run_job(args,group,seed,parts,bert,inputs):
    folder=args.out/group/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    phases=[1,1,1] if args.smoke else [5,8,7]
    signature=dict(group=group,seed=seed,phases=phases,batch=args.batch,acc_steps=args.acc_steps,
                   inputs=inputs,torch=torch.__version__,transformers=transformers.__version__,numpy=np.__version__,
                   fp32=True,cudnn=False,tf32=False,paired_loader_rng=True,select='global_validation_TAV_macro_f1',
                   loss='CE(label_smoothing=.05,class_weight=sqrt_inverse)+.02*NCE+.02*OT+.001*legacy_reg',
                   modality_dropout=.15,auxiliary_recipe='unchanged_legacy',test_evaluated=False)
    signature=json.loads(json.dumps(signature));complete=folder/'complete.json'
    if complete.exists():
        old=read_json(complete)
        if old['signature']!=signature: raise RuntimeError('Completed training configuration changed')
        for filename,digest in old['artifacts'].items():
            if sha_file(folder/filename)!=digest: raise RuntimeError(f'Completed artifact changed: {filename}')
        print('PAPER_AUDIO_TRAIN_CACHE',group,seed,flush=True);return
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    resume_file=folder/'resume.pt';resume=None
    if resume_file.exists():
        # This file is locally generated optimizer/RNG state, not a downloaded checkpoint.
        resume=torch.load(resume_file,map_location='cpu',weights_only=False)
        if resume['signature']!=signature: raise RuntimeError('Resume configuration changed')
    tr,va=parts['train'],parts['valid']
    model=make_model(bert,tr['audio'].shape[1],tr['vision'].shape[1],pretrained=resume is None).to(args.device)
    identity=dict(parameter_count=sum(p.numel() for p in model.parameters()),random_initialization_sha256=initial_digest(model),d_model=model.d)
    identity_file=folder/'model_identity.json'
    if resume is None: record_json(identity_file,identity)
    else:
        identity=read_json(identity_file);model.load_state_dict(resume['model'],strict=True)
    paired=args.out/'initialization'/f'seed_{seed}.json';record_json(paired,identity)
    generator=torch.Generator().manual_seed(seed);train_loader=loader(tr,args.batch,True,generator)
    counts=np.bincount(tr['y'],minlength=2)
    if min(counts)==0: raise ValueError('Both classes required in training')
    weights=np.sqrt(counts.sum()/(2*counts));weights=weights/weights.mean()
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=args.device),label_smoothing=.05)
    history=resume['history'] if resume else [];best=resume['best'] if resume else dict(val_macro_f1=-1.)
    started=time.monotonic()
    print('PAPER_AUDIO_TRAIN_START',group,seed,'phases',phases,'n_train',len(tr['y']),'n_valid',len(va['y']),flush=True)
    for phase,epochs in enumerate(phases):
        if resume and phase<resume['phase']: continue
        if phase==0: legacy.freeze_bert_all(model);lr_b=0.;lr_p=3e-4
        elif phase==1: legacy.unfreeze_bert_last_n(model,2,verbose=False);lr_b=2e-5;lr_p=5e-5
        else: legacy.unfreeze_bert_all(model,verbose=False);lr_b=1e-5;lr_p=5e-5
        optimizer=legacy.build_optimizer(model,lr_b,lr_p,3e-4)
        steps=math.ceil(len(train_loader)/args.acc_steps);total=steps*epochs
        scheduler=get_cosine_schedule_with_warmup(optimizer,max(1,int(total*.1)),total)
        first=1;phase_best=-1.;phase_file=folder/f'phase_best_{phase}.pt'
        if resume and phase==resume['phase']:
            optimizer.load_state_dict(resume['optimizer']);scheduler.load_state_dict(resume['scheduler'])
            restore_rng(resume['rng']);generator.set_state(resume['generator']);first=resume['epoch']+1;phase_best=resume['phase_best']
        for epoch in range(first,epochs+1):
            tick=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.
            for i,(tb,ta,tv,y) in enumerate(train_loader):
                tb,ta,tv,y=tb.to(args.device),ta.to(args.device),tv.to(args.device),y.long().to(args.device)
                logits,aux=model(tb[:,0],tb[:,1],tb[:,2],ta,tv,training=True)
                loss=criterion(logits,y)+.02*aux['gcl']+.02*aux['ot']+.001*aux['reg_ama']
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                group_start=(i//args.acc_steps)*args.acc_steps;actual=min(args.acc_steps,len(train_loader)-group_start)
                (loss/actual).backward();loss_sum+=float(loss.detach().cpu())
                if (i+1)%args.acc_steps==0 or i+1==len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
            _,metrics,_=evaluate(model,va,max(8,args.batch),args.device);val=metrics['TAV']
            if val['macro_f1']>phase_best:
                phase_best=val['macro_f1'];save_torch(phase_file,cpu_state(model))
            if val['macro_f1']>best['val_macro_f1']:
                best=dict(val_macro_f1=val['macro_f1'],val_accuracy=val['accuracy'],phase=phase+1,epoch=epoch)
                save_torch(folder/'best.pt',cpu_state(model));dump_json(folder/'best_validation.json',best)
            history.append(dict(phase=phase+1,epoch=epoch,train_loss=loss_sum/len(train_loader),val_macro_f1=val['macro_f1'],
                                val_accuracy=val['accuracy'],seconds=round(time.monotonic()-tick,2)))
            write_csv(folder/'validation_history.csv',history)
            save_torch(resume_file,dict(signature=signature,phase=phase,epoch=epoch,phase_best=phase_best,best=best,
                       model=cpu_state(model),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                       rng=rng_state(),generator=generator.get_state(),history=history))
            dump_json(args.out/'current_job.json',dict(group=group,seed=seed,phase=phase+1,epoch=epoch,phase_epochs=epochs,
                       latest=history[-1],best_validation=best,status='training',test_evaluated=False))
            print('PAPER_AUDIO_EPOCH',group,seed,'P',phase+1,'E',epoch,'val_f1',round(val['macro_f1'],6),'seconds',history[-1]['seconds'],flush=True)
        model.load_state_dict(torch.load(phase_file,map_location='cpu',weights_only=True),strict=True)
        resume=None;del optimizer,scheduler;gc.collect()
    model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True),strict=True)
    output,metrics,checks=evaluate(model,va,max(8,args.batch),args.device,all_masks=True)
    if abs(metrics['TAV']['macro_f1']-best['val_macro_f1'])>1e-10: raise RuntimeError('Best checkpoint replay mismatch')
    atomic_npz(folder/'validation_predictions.npz',ids=va['ids'],labels=va['y'],**output)
    dump_json(folder/'validation_metrics.json',dict(metrics=metrics,checks=checks,test_evaluated=False))
    files=['best.pt','best_validation.json','model_identity.json','validation_history.csv','validation_predictions.npz','validation_metrics.json']
    dump_json(complete,dict(signature=signature,best_validation=best,identity=identity,checks=checks,
              artifacts={f:sha_file(folder/f) for f in files},elapsed_seconds=round(time.monotonic()-started,2),status='complete',test_evaluated=False))
    resume_file.unlink(missing_ok=True)
    for phase in range(len(phases)): (folder/f'phase_best_{phase}.pt').unlink(missing_ok=True)
    print('PAPER_AUDIO_TRAIN_DONE',group,seed,'val_f1',best['val_macro_f1'],'test_evaluated=False',flush=True)
    del model;gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=HERE.parent.parent)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--batch',type=int,default=8);ap.add_argument('--acc-steps',type=int,default=4)
    ap.add_argument('--seeds',type=int,nargs='+',default=[42]);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();verify_code();a.device=runtime(a.device,4)
    if a.batch<2 or a.acc_steps<1: raise ValueError('batch>=2, acc_steps>=1 required')
    plan=read_json(a.out/'inventory.json')
    if plan['smoke']!=a.smoke: raise RuntimeError('Smoke and pilot data cannot be mixed')
    from paper_data import load_parts
    parts,embeds=load_parts(a.out);groups=representations(parts,embeds,a.out)
    inputs={f:sha_file(a.out/f) for f in ['inventory.json','bert_verified.json','extraction_signature.json',
            'media_manifest.json','paper_train.npz','paper_valid.npz','embedding_train.npz','embedding_valid.npz','representation_stats.npz']}
    inputs['bundle_manifest_sha256']=sha_file(HERE/'bundle_manifest.json')
    bert=a.root/'MSA Datasets/pretrained_berts/bert_en';done=0
    for group in GROUPS:
        for seed in a.seeds:
            run_job(a,group,seed,groups[group],bert,inputs);done+=1
            dump_json(a.out/'suite_progress.json',dict(done=done,total=len(GROUPS)*len(a.seeds),status='running',test_evaluated=False))
    dump_json(a.out/'suite_progress.json',dict(done=done,total=done,status='complete',test_evaluated=False))
    print('PAPER_AUDIO_TRAIN_SUITE_COMPLETE',done,'jobs',flush=True)
if __name__=='__main__': main()
