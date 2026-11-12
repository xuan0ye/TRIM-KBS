"""Verify exact datasets/weights and equivalent BERT configuration parameters."""
from pathlib import Path
import argparse,hashlib,inspect,json,shutil
from common import sha_file,dump_json
from runtime_v2 import runtime
from bert_weight_check import compare_weights
import torch,transformers,numpy as np
HERE=Path(__file__).resolve().parent
CONFIG_METADATA={"_name_or_path","_commit_hash","transformers_version","architectures"}

def read_config(path):
 def pairs(items):
  result={}
  for key,value in items:
   if key in result:raise ValueError(f"Duplicate JSON configuration key: {key}")
   result[key]=value
  return result
 def reject(value):raise ValueError(f"Nonfinite JSON value: {value}")
 value=json.loads(Path(path).read_text(encoding="utf-8-sig"),object_pairs_hook=pairs,parse_constant=reject)
 if not isinstance(value,dict):raise ValueError("BERT config must be a JSON object")
 return value

def config_parameters(value):
 raw=dict(value)
 if raw.get("model_type","bert")!="bert":raise ValueError(f"Expected BERT model_type, got {raw.get('model_type')!r}")
 dtype_values=[str(raw[k]).replace("torch.","") for k in ("dtype","torch_dtype") if raw.get(k) is not None]
 if len(set(dtype_values))>1:raise ValueError("Conflicting dtype and torch_dtype values")
 dtype=dtype_values[0] if dtype_values else "float32"
 raw.pop("dtype",None);raw.pop("torch_dtype",None)
 checkpointing=raw.pop("gradient_checkpointing",False)
 # Resolve omitted defaults using the SAME installed BertConfig for both sides.
 # Preserve unknown/custom fields so a potentially meaningful change cannot pass silently.
 parsed=transformers.BertConfig.from_dict(raw).to_dict()
 for key in CONFIG_METADATA|{"dtype","torch_dtype"}:parsed.pop(key,None)
 parsed["effective_dtype"]=dtype;parsed["gradient_checkpointing"]=checkpointing
 return parsed

def compare_config(reference,actual):
 left=config_parameters(reference);right=config_parameters(actual)
 missing={"missing":True}
 differences={k:{"expected":left.get(k,missing),"actual":right.get(k,missing)} for k in sorted(set(left)|set(right)) if left.get(k,missing)!=right.get(k,missing)}
 canonical=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf8")).hexdigest()
 return differences,canonical(left),canonical(right)

def verify_input(root,entry):
 path=Path(root)/entry["relative_path"]
 result=dict(relative_path=entry["relative_path"],expected_sha256=entry["sha256"],status="failed")
 if not path.is_file() and entry.get("kind")=="bert_weights":
  fallback=entry.get("fallback_relative_path")
  if fallback and (Path(root)/fallback).is_file():path=Path(root)/fallback
 result["selected_relative_path"]=str(path.relative_to(Path(root)))
 if not path.is_file():
  extra=f" (also checked {entry.get('fallback_relative_path')})" if entry.get("kind")=="bert_weights" else ""
  return result,f"Required input missing: {path}{extra}"
 actual=sha_file(path);result["actual_sha256"]=actual
 if actual==entry["sha256"]:
  result.update(status="passed",verification="exact_sha256");return result,None
 if entry.get("kind")=="bert_weights" and isinstance(entry.get("tensor_reference"),dict):
  try:comparison=compare_weights(path,entry["tensor_reference"])
  except Exception as exc:return result,f"Cannot verify BERT tensors {path}: {exc}"
  result["tensor_comparison"]=comparison
  if comparison["passed"]:
   result.update(status="passed",verification="exact_bert_tensors")
   print("VERIFIED_BERT_TENSORS",path,f"matched={comparison['matched_tensors']}/{comparison['total_reference_tensors']}",flush=True)
   return result,None
  short={k:v for k,v in comparison.items() if k not in ("mismatched","ignored_pretraining_heads")}
  short["mismatched_tensor_names"]=list(comparison["mismatched"])
  return result,f"BERT tensor mismatch: {path}\n{json.dumps(short,ensure_ascii=False,indent=2)}"
 if entry.get("kind")=="bert_config" and isinstance(entry.get("config_reference"),dict):
  try:diff,expected_parameters,actual_parameters=compare_config(entry["config_reference"],read_config(path))
  except Exception as exc:return result,f"Invalid BERT configuration {path}: {exc}"
  result.update(expected_parameters_sha256=expected_parameters,actual_parameters_sha256=actual_parameters,differences=diff)
  if not diff:
   result.update(status="passed",verification="equivalent_bert_parameters")
   print("VERIFIED_CONFIG_EQUIVALENT",path,"(file bytes differ; parsed parameters match)",flush=True)
   return result,None
  return result,f"BERT parameter mismatch: {path}\n{json.dumps(diff,ensure_ascii=False,indent=2)}"
 return result,f"SHA256 differs: {path}\nExpected {entry['sha256']}\nActual   {actual}"

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=HERE.parent);ap.add_argument("--datasets",nargs="+",default=["MOSI","SIMS"],choices=["MOSI","SIMS","MOSEI"]);ap.add_argument("--view",choices=["native","aligned"],default="native");ap.add_argument("--require-cuda",action="store_true");args=ap.parse_args()
 device=runtime();bundle=HERE/"bundle_manifest.json"
 if bundle.exists():
  for name,expected in json.loads(bundle.read_text(encoding="utf8"))["files"].items():
   if sha_file(HERE/name)!=expected:raise RuntimeError(f"Upload file differs: {name}")
 if "mmap" not in inspect.signature(torch.load).parameters:raise RuntimeError("This runner requires PyTorch with mmap checkpoint loading (2.1 or later).")
 if args.require_cuda and device.type!="cuda":raise RuntimeError("No CUDA GPU visible. Check CUDA_VISIBLE_DEVICES and the activated environment.")
 results=[];errors=[]
 for entry in json.loads((HERE/"input_manifest.json").read_text(encoding="utf8"))["inputs"]:
  if not set(entry["datasets"])&set(args.datasets) or args.view not in entry["views"]:continue
  print("VERIFY_INPUT",args.root/entry["relative_path"],flush=True)
  result,error=verify_input(args.root,entry);results.append(result)
  if error:errors.append(error)
 report=HERE/"logs/preflight_inputs.json"
 dump_json(report,dict(status="failed" if errors else "passed",datasets=args.datasets,view=args.view,torch=torch.__version__,transformers=transformers.__version__,inputs=results,errors=errors))
 if errors:raise RuntimeError("Input verification failed:\n\n"+"\n\n".join(errors)+f"\n\nFull report: {report}")
 for d in args.datasets:
  with np.load(HERE/f"reference_ids/{d}.npz",allow_pickle=False) as z:
   assert len(set(z["ids"].tolist()))==len(z["labels"]) and np.isin(z["labels"],[0,1]).all()
 assert not torch.backends.cudnn.enabled
 print("PREFLIGHT_PASSED",dict(torch=torch.__version__,transformers=transformers.__version__,device=str(device),cudnn_enabled=torch.backends.cudnn.enabled,free_disk_GB=round(shutil.disk_usage(HERE).free/1e9,1)),flush=True)
 if device.type=="cuda":print("GPU",torch.cuda.get_device_name(0),flush=True)
if __name__=="__main__":main()
