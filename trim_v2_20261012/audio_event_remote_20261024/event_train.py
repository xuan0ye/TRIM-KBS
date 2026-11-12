"""Frozen-parent, equal-capacity audio event pilot, validation only."""
from pathlib import Path
import argparse
import gc
import math
import subprocess
import sys
import time
import numpy as np
import torch
import transformers
from event_common import *
from event_model import EventCorrection
from data_v2 import SequenceDataset, train_statistics, make_loader, to_device
from runtime_v2 import runtime, seed_everything, rng_state, restore_rng, save_torch
from model_v2 import TRIMV2
from aggregate_core import scores

VARIANTS = ("embedding", "events", "events_shuffled")
INPUT_KEYS = ("base_logits", "text", "audio", "visual", "audio_present", "visual_present")


def require_extraction(out):
    complete = read_json(out/"extraction_complete.json")
    if complete["status"] != "complete" or complete["test_evaluated"]:
        raise RuntimeError("Invalid extraction completion record.")
    for name, digest in complete["files"].items():
        if sha_file(out/name) != digest:
            raise RuntimeError(f"Extraction output changed: {name}")
    return complete


def verified_parent(args, stats, data_sha):
    candidate = args.parent
    if candidate is None and not args.smoke:
        existing = args.root/"trim_v2_20261012/experiments/MOSEI/mean/seed_42"
        if (existing/"training_complete.json").is_file():
            candidate = existing
    if candidate is None:
        destination = args.out/"baseline_source"
        command = [sys.executable, "-u", str(CORE/"train_v2.py"), "--root", str(args.root),
                   "--datasets", "MOSEI", "--variants", "mean", "--seeds", "42",
                   "--out", str(destination), "--device", str(args.device),
                   "--batch", "8", "--acc-steps", "4"]
        if args.smoke:
            command.append("--smoke")
        subprocess.run(command, check=True)
        candidate = destination/"MOSEI/mean/seed_42"
    candidate = candidate.resolve()
    metadata = read_json(candidate/"training_complete.json")
    sig = metadata["signature"]
    if metadata["status"] != "complete" or (sig["dataset"],sig["variant"],sig["seed"],sig["view"]) != ("MOSEI","mean",42,"native"):
        raise RuntimeError("Parent must be a completed native-view MOSEI mean seed-42 model.")
    if not args.smoke and sig["smoke"]:
        raise RuntimeError("A smoke checkpoint cannot be a pilot baseline.")
    if sig["data_sha256"] != data_sha or sig["train_statistics"] != stats:
        raise RuntimeError("Parent input data or full-training feature statistics differ.")
    for file, digest in sig["code_sha256"].items():
        if sha_file(CORE/file) != digest:
            raise RuntimeError(f"Parent source differs from the bundled core: {file}")
    checkpoint = candidate/"best.pt"
    if sha_file(checkpoint) != metadata["checkpoint_sha256"]:
        raise RuntimeError("Parent checkpoint checksum mismatch.")
    weight = args.root/"MSA Datasets/pretrained_berts/bert_en/model.safetensors"
    if not weight.is_file():
        weight = weight.with_name("pytorch_model.bin")
    if sha_file(weight) != sig["bert_weights_sha256"]:
        raise RuntimeError("Parent BERT weight source differs.")
    source = dict(directory=str(candidate), checkpoint_sha256=metadata["checkpoint_sha256"],
                  metadata_sha256=sha_file(candidate/"training_complete.json"),
                  source_signature=sig, best_validation=metadata["best_validation"],
                  test_evaluated_by_this_pilot=False)
    dump_json(args.out/"parent_source.json", source)
    return checkpoint, source


def cache_parent(args, parts, stats, checkpoint, source):
    signature = dict(checkpoint_sha256=source["checkpoint_sha256"],
                     data_sha256=source["source_signature"]["data_sha256"],
                     smoke=args.smoke, core_sha256=sha_file(CORE/"model_v2.py"),
                     cache_code_sha256=sha_file(Path(__file__)),
                     torch=torch.__version__, transformers=transformers.__version__)
    complete_path = args.out/"parent_cache_complete.json"
    if complete_path.is_file():
        record = read_json(complete_path)
        if record["signature"] != signature:
            raise RuntimeError("Parent cache settings changed.")
        for name, digest in record["files"].items():
            if sha_file(args.out/name) != digest:
                raise RuntimeError(f"Parent cache changed: {name}")
        return
    bert = args.root/"MSA Datasets/pretrained_berts/bert_en"
    model = TRIMV2(bert,parts["train"].a_dim,parts["train"].v_dim,"mean",pretrained=False)
    model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True,mmap=True),strict=True)
    for key, encoder in (("audio",model.audio),("vision",model.visual)):
        for field in ("mean","std"):
            torch.testing.assert_close(getattr(encoder,"feature_"+field),
                                       torch.tensor(stats[key][field],dtype=torch.float32),atol=0,rtol=0)
    model = model.to(args.device).eval().requires_grad_(False)
    files = {}
    with torch.inference_mode():
        for split, ds in parts.items():
            saved = {k:[] for k in INPUT_KEYS}
            for number, batch in enumerate(make_loader(ds,16)):
                rep = model.representations(to_device(batch,args.device))
                component = model.components_from_repr(rep)
                values = dict(base_logits=model.all_patterns(component),text=rep["text"],
                              audio=rep["audio"].mean(1),visual=rep["visual"].mean(1),
                              audio_present=rep["a_present"],visual_present=rep["v_present"])
                for key, value in values.items():
                    saved[key].append(value.cpu().numpy())
                if number % 100 == 0:
                    print("EVENT_PARENT_CACHE",split,number*16,"/",len(ds),flush=True)
            saved = {k:np.concatenate(v) for k,v in saved.items()}
            if not np.isfinite(saved["base_logits"]).all():
                raise ValueError("Nonfinite parent logits.")
            if split == "valid" and not args.smoke:
                reproduced = metric_report(ds.y,saved["base_logits"])
                if abs(reproduced["selection_score"]-source["best_validation"]["selection_score"]) > 1e-12:
                    raise RuntimeError("Parent validation cache does not reproduce its selected score.")
            name = f"parent_{split}.npz"
            atomic_npz(args.out/name,ids=ds.ids,labels=ds.y,**saved)
            files[name] = sha_file(args.out/name)
    dump_json(complete_path,dict(signature=signature,files=files,test_evaluated=False))
    del model
    gc.collect()
    if args.device.type == "cuda":
        torch.cuda.empty_cache()


def as_tensors(cache, features, device):
    return {**{k:torch.from_numpy(cache[k]).to(device) for k in INPUT_KEYS},
            "event":torch.from_numpy(features).to(device),
            "y":torch.tensor(cache["labels"],dtype=torch.long,device=device)}


def forward(model, data, indices=None, event_present=None):
    data = {k:(v if indices is None else v[indices]) for k,v in data.items()}
    return model(**{k:data[k] for k in INPUT_KEYS},event=data["event"],event_present=event_present)


def metric_report(labels, logits):
    logits = np.asarray(logits,dtype=np.float64)
    shifted = logits-logits.max(-1,keepdims=True)
    prob = np.exp(shifted);prob /= prob.sum(-1,keepdims=True)
    conditions = {k:scores(labels,logits[:,i].argmax(-1),prob[:,i,1]) for i,k in enumerate(PATTERNS)}
    return dict(selection_score=float(np.mean([r["macro_f1"] for r in conditions.values()])),
                conditions=conditions)


def validation_logits(model, data):
    model.eval()
    with torch.inference_mode():
        logits = forward(model,data)
    if not torch.isfinite(logits).all():
        raise ValueError("Nonfinite adapted logits.")
    torch.testing.assert_close(logits[:,(0,2)],data["base_logits"][:,(0,2)],atol=0,rtol=0)
    return logits.cpu().numpy()


def diagnostics(model, data, cache, path):
    model.eval()
    with torch.inference_mode():
        real = forward(model,data)
        disabled = forward(model,data,event_present=torch.zeros_like(data["audio_present"]))
        torch.testing.assert_close(disabled,data["base_logits"],atol=0,rtol=0)
        shuffles, indices = [], []
        for seed in (13,21,42):
            permutation = derangement(len(cache["ids"]),seed)
            changed = dict(data,event=data["event"][torch.tensor(permutation,device=data["event"].device)])
            prediction = forward(model,changed)
            torch.testing.assert_close(prediction[:,(0,2)],real[:,(0,2)],atol=0,rtol=0)
            shuffles.append(prediction.cpu().numpy());indices.append(permutation)
        missing = dict(data,base_logits=data["base_logits"].clone(),
                       audio_present=torch.zeros_like(data["audio_present"]))
        missing["base_logits"][:,1] = missing["base_logits"][:,0]
        missing["base_logits"][:,3] = missing["base_logits"][:,2]
        missing_logits = forward(model,missing)
        torch.testing.assert_close(missing_logits,missing["base_logits"],atol=0,rtol=0)
    atomic_npz(path,ids=cache["ids"],labels=cache["labels"],logits=real.cpu().numpy(),
               disabled_logits=disabled.cpu().numpy(),shuffle_logits=np.stack(shuffles),
               shuffle_indices=np.stack(indices),audio_missing_logits=missing_logits.cpu().numpy())


def run_adapter(args, variant, train, valid, train_features, valid_features, signature):
    folder = args.out/"adapters"/variant/"seed_42"
    folder.mkdir(parents=True,exist_ok=True)
    source_signature = dict(**signature,variant=variant,seed=SEED,epochs=args.epochs,
                            batch=args.batch,projection_dim=128,
                            optimizer="AdamW lr=3e-4 weight_decay=.01; cosine per epoch; clip=1",
                            loss="equal four-condition CE; smoothing=.05; train-only sqrt-inverse-frequency weights",
                            selection="validation four-condition Macro-F1 mean, epoch 0 included")
    if (folder/"complete.json").is_file():
        record = read_json(folder/"complete.json")
        if record["signature"] != source_signature:
            raise RuntimeError(f"Adapter settings changed: {folder}")
        for name,digest in record["files"].items():
            if sha_file(folder/name) != digest:
                raise RuntimeError(f"Adapter output changed: {folder/name}")
        print("VERIFIED_EVENT_ADAPTER",variant,flush=True)
        return
    seed_everything(SEED)
    model = EventCorrection().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    generator = torch.Generator().manual_seed(SEED)
    tr = as_tensors(train,train_features,args.device)
    va = as_tensors(valid,valid_features,args.device)
    counts = np.bincount(train["labels"],minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both sentiment classes required.")
    weights = np.sqrt(counts.sum()/(2*counts));weights /= weights.mean()
    weights = torch.tensor(weights,dtype=torch.float32,device=args.device)
    history, first, began = [], 1, time.monotonic()
    initial = metric_report(valid["labels"],valid["base_logits"])
    resume_path = folder/"resume.pt"
    if resume_path.is_file():
        saved = torch.load(resume_path,map_location="cpu",weights_only=False)
        if saved["signature"] != source_signature:
            raise RuntimeError("Resume signature differs.")
        model.load_state_dict(saved["model"]);optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"]);restore_rng(saved["rng"],generator)
        first = saved["epoch"]+1;best = saved["best"];history = saved["history"]
    else:
        torch.testing.assert_close(torch.from_numpy(validation_logits(model,va)),
                                   torch.from_numpy(valid["base_logits"]),atol=0,rtol=0)
        best = dict(epoch=0,**initial)
        save_torch(folder/"best.pt",{k:v.cpu() for k,v in model.state_dict().items()})
    for epoch in range(first,args.epochs+1):
        model.train();tick = time.monotonic();total_loss = 0.
        order = torch.randperm(len(train["ids"]),generator=generator)
        for start in range(0,len(order),args.batch):
            idx = order[start:start+args.batch].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward(model,tr,idx)
            loss = sum(torch.nn.functional.cross_entropy(logits[:,i],tr["y"][idx],
                       weight=weights,label_smoothing=.05) for i in range(4))/4
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite adapter loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();total_loss += float(loss.detach().cpu())
        scheduler.step()
        val = metric_report(valid["labels"],validation_logits(model,va))
        if val["selection_score"] > best["selection_score"]:
            best = dict(epoch=epoch,**val)
            save_torch(folder/"best.pt",{k:v.cpu() for k,v in model.state_dict().items()})
        history.append(dict(epoch=epoch,loss=total_loss/math.ceil(len(order)/args.batch),
                            selection_score=val["selection_score"],seconds=time.monotonic()-tick))
        write_csv(folder/"history.csv",history)
        save_torch(resume_path,dict(signature=source_signature,epoch=epoch,best=best,history=history,
                   model={k:v.cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(),rng=rng_state(generator)))
        dump_json(args.out/"current_job.json",dict(variant=variant,epoch=epoch,epochs=args.epochs,
                  best_validation_score=best["selection_score"],selected_epoch=best["epoch"],status="training"))
        print("EVENT_EPOCH",variant,f"{epoch}/{args.epochs}","val_mean_f1",round(val["selection_score"],6),flush=True)
        if getattr(args,"stop_after_epoch",0) and epoch >= args.stop_after_epoch:
            return "paused"
    model.load_state_dict(torch.load(folder/"best.pt",map_location="cpu",weights_only=True))
    diagnostics(model,va,valid,folder/"validation_predictions.npz")
    chosen = metric_report(valid["labels"],load_npz(folder/"validation_predictions.npz")["logits"])
    if abs(chosen["selection_score"]-best["selection_score"]) > 1e-12:
        raise RuntimeError("Chosen checkpoint does not reproduce validation.")
    dump_json(folder/"best_validation.json",best)
    dump_json(folder/"complete.json",dict(status="complete",signature=source_signature,
              test_evaluated=False,initial_selection_score=initial["selection_score"],
              selected_epoch=best["epoch"],selected_epoch_zero=best["epoch"]==0,
              training_gain_pp=100*(best["selection_score"]-initial["selection_score"]),
              trainable_parameters=sum(p.numel() for p in model.parameters()),
              elapsed_seconds=time.monotonic()-began,
              files={name:sha_file(folder/name) for name in
                     ("best.pt","history.csv","validation_predictions.npz","best_validation.json")}))
    resume_path.unlink(missing_ok=True)
    print("EVENT_ADAPTER_DONE",variant,"selected_epoch",best["epoch"],flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=HERE.parents[1])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--parent",type=Path)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--batch",type=int,default=128)
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--smoke",action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 2:
        parser.error("epochs>=1 and batch>=2 required")
    if args.smoke:
        args.epochs = 2
    args.device = runtime(args.device,4)
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable.")
    extraction = require_extraction(args.out)
    plan = read_json(args.out/"inventory.json")
    if plan.get("probe",False) or set(plan["splits"]) != {"train","valid"}:
        raise RuntimeError("A training-media probe cannot be used for the full comparison.")
    if plan["smoke"] != args.smoke or plan["data_sha256"] != sha_file(data_file(args.root)):
        raise RuntimeError("Extraction data/smoke mode differs.")
    raw = load_raw(args.root)
    full = SequenceDataset(raw,"MOSEI","train","native")
    stats = train_statistics(full)
    parts = {s:SequenceDataset(raw,"MOSEI",s,"native",limit=16 if args.smoke else None)
             for s in ("train","valid")}
    checkpoint, source = verified_parent(args,stats,plan["data_sha256"])
    cache_parent(args,parts,stats,checkpoint,source)
    caches, events = {}, {}
    for split in ("train","valid"):
        caches[split] = load_npz(args.out/f"parent_{split}.npz")
        events[split] = load_npz(args.out/f"events_{split}.npz")
        expected_ids = np.asarray([r["id"] for r in plan["splits"][split]])
        if not np.array_equal(parts[split].ids,expected_ids) or not np.array_equal(caches[split]["ids"],expected_ids) or not np.array_equal(events[split]["ids"],expected_ids):
            raise RuntimeError(f"Parent/event/sample IDs or order differ: {split}")
        if not np.array_equal(caches[split]["labels"],parts[split].y):
            raise RuntimeError("Cached labels differ.")
    signature = dict(release=RELEASE,dataset="MOSEI",smoke=args.smoke,test_evaluated=False,
                     parent_checkpoint_sha256=source["checkpoint_sha256"],
                     extraction_manifest_sha256=sha_file(args.out/"extraction_complete.json"),
                     parent_cache_sha256=sha_file(args.out/"parent_cache_complete.json"),
                     model_sha256=sha_file(HERE/"event_model.py"),
                     train_sha256=sha_file(Path(__file__)),common_sha256=sha_file(HERE/"event_common.py"),
                     torch=torch.__version__,transformers=transformers.__version__,
                     protocol="Frozen mean parent; fixed random 128D projections, fitted train-only scaling; identical adapters; TA/TAV corrections only.")
    dump_json(args.out/"training_protocol.json",signature)
    projected = {}
    for kind in ("events","embedding"):
        tr,va,projection = project_features(events["train"][kind],events["valid"][kind])
        atomic_npz(args.out/f"projection_{kind}.npz",**projection)
        projected[kind] = (tr,va)
    tr_shuffle = derangement(len(caches["train"]["ids"]),20261024)
    va_shuffle = derangement(len(caches["valid"]["ids"]),20261025)
    atomic_npz(args.out/"training_shuffle_indices.npz",train=tr_shuffle,valid=va_shuffle,
               train_ids=caches["train"]["ids"],valid_ids=caches["valid"]["ids"])
    dump_json(args.out/"training_progress.json",dict(status="running",done=0,total=3))
    for number,variant in enumerate(VARIANTS,1):
        kind = "embedding" if variant == "embedding" else "events"
        tr,va = projected[kind]
        if variant == "events_shuffled":
            tr,va = tr[tr_shuffle],va[va_shuffle]
        run_adapter(args,variant,caches["train"],caches["valid"],tr,va,signature)
        dump_json(args.out/"training_progress.json",dict(status="running",done=number,total=3,last=variant))
    dump_json(args.out/"training_progress.json",dict(status="complete",done=3,total=3,test_evaluated=False))
    print("EVENT_TRAINING_COMPLETE","3 adapters plus baseline","test_evaluated",False,flush=True)


if __name__ == "__main__":
    main()
