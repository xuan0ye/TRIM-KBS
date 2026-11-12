"""Verify the BERT tensors used by BertModel across supported checkpoint containers."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import torch

@contextmanager
def weight_reader(path):
 path=Path(path)
 if path.suffix==".safetensors":
  from safetensors import safe_open
  with safe_open(str(path),framework="pt",device="cpu") as reader:yield tuple(reader.keys()),reader.get_tensor
 elif path.name=="pytorch_model.bin":
  # Model files supplied by the user; restricted loading never executes arbitrary pickle globals.
  state=torch.load(path,map_location="cpu",weights_only=True)
  if not isinstance(state,dict) or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
   raise ValueError("Expected a plain Hugging Face BERT state dictionary in pytorch_model.bin")
  yield tuple(state),state.__getitem__
 else:raise ValueError(f"Unsupported BERT checkpoint container: {path.name}")

def tensor_record(tensor):
 if tensor.layout!=torch.strided or tensor.is_quantized:raise ValueError("Expected dense, unquantized BERT tensors")
 value=tensor.detach().cpu().contiguous()
 raw=value.reshape(-1).view(torch.uint8).numpy().tobytes()
 return dict(shape=list(value.shape),dtype=str(value.dtype).replace("torch.",""),sha256=hashlib.sha256(raw).hexdigest())

def make_reference(path):
 with weight_reader(path) as (names,get):
  return {name:tensor_record(get(name)) for name in sorted(names)}

def canonical_bert_name(name):
 # Match Transformers 4.49 PreTrainedModel._fix_state_dict_key_on_load.
 if name.startswith("bert."):name=name[5:]
 if name.endswith("LayerNorm.beta"):return name[:-len("beta")]+"bias"
 if name.endswith("LayerNorm.gamma"):return name[:-len("gamma")]+"weight"
 return name

def compare_weights(path,reference):
 normalized_reference={};reference_aliases=0
 for original,record in reference.items():
  canonical=canonical_bert_name(original)
  if canonical in normalized_reference:raise ValueError(f"Ambiguous reference tensor aliases: {canonical}")
  normalized_reference[canonical]=record
  if canonical!=original:reference_aliases+=1
 reference=normalized_reference
 with weight_reader(path) as (names,get):
  mapped={};ignored=[];ignored_buffers=[];bad_buffers=[];unexpected=[];duplicates=[]
  # BertModel.from_pretrained recognizes the standard base_model_prefix "bert".
  checkpoint_aliases=0
  for name in names:
   canonical=canonical_bert_name(name)
   if name.endswith(("LayerNorm.beta","LayerNorm.gamma")):checkpoint_aliases+=1
   if canonical in reference:
    if canonical in mapped:duplicates.append(canonical)
    mapped[canonical]=name
   elif canonical in ("embeddings.position_ids","embeddings.token_type_ids") and "embeddings.position_embeddings.weight" in reference:
    # Older BERT checkpoints may serialize these deterministic embedding buffers.
    # Accept them only if they equal the buffers constructed from the validated configuration.
    length=reference["embeddings.position_embeddings.weight"]["shape"][0]
    expected=torch.arange(length,dtype=torch.long)[None] if canonical.endswith("position_ids") else torch.zeros((1,length),dtype=torch.long)
    observed=get(name)
    if observed.dtype==expected.dtype and torch.equal(observed,expected):ignored_buffers.append(name)
    else:bad_buffers.append(name)
   elif name.startswith("cls."):
    # Masked-language/next-sentence pretraining heads are unused by explicit BertModel.
    ignored.append(name)
   else:unexpected.append(name)
  result=dict(reference_aliases_normalized=reference_aliases,checkpoint_layernorm_aliases_normalized=checkpoint_aliases,missing=sorted(set(reference)-set(mapped)),unexpected=sorted(unexpected),duplicate_names=sorted(duplicates),mismatched={},ignored_pretraining_heads=sorted(ignored),verified_embedding_buffers=sorted(ignored_buffers),invalid_embedding_buffers=sorted(bad_buffers),matched_tensors=0,total_reference_tensors=len(reference))
  for canonical in sorted(mapped):
   observed=tensor_record(get(mapped[canonical]));expected=reference[canonical]
   if observed==expected:result["matched_tensors"]+=1
   else:result["mismatched"][canonical]=dict(expected=expected,actual=observed)
  result["passed"]=not any((result["missing"],result["unexpected"],result["duplicate_names"],result["mismatched"],result["invalid_embedding_buffers"]))
  return result
