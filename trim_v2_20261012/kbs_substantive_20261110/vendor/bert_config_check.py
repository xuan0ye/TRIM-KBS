"""Compare effective BertModel configuration, ignoring serialization metadata only.

Adapted from the project's existing preflight_v2.py equivalence checks.
Both sides are resolved with the same installed BertConfig, not cross-version defaults.
"""
from pathlib import Path
import hashlib
import json
import transformers

CONFIG_METADATA = {"_name_or_path", "_commit_hash", "transformers_version", "architectures"}


def read_config(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON configuration key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    value = json.loads(Path(path).read_text(encoding="utf-8-sig"),
                       object_pairs_hook=pairs, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("BERT config must be a JSON object")
    # Also reject overflowed numeric literals such as 1e999.
    json.dumps(value, allow_nan=False)
    return value


def config_parameters(value):
    raw = dict(value)
    if raw.get("model_type", "bert") != "bert":
        raise ValueError(f"Expected BERT model_type, got {raw.get('model_type')!r}")
    dtypes = [str(raw[k]).replace("torch.", "") for k in ("dtype", "torch_dtype")
              if raw.get(k) is not None]
    if len(set(dtypes)) > 1:
        raise ValueError("Conflicting dtype and torch_dtype values")
    dtype = dtypes[0] if dtypes else "float32"
    raw.pop("dtype", None)
    raw.pop("torch_dtype", None)
    checkpointing = raw.pop("gradient_checkpointing", False)
    # BertModel is constructed explicitly; these fields describe the saved file,
    # not the architecture used by this experiment.
    parsed = transformers.BertConfig.from_dict(raw).to_dict()
    for key in CONFIG_METADATA | {"dtype", "torch_dtype"}:
        parsed.pop(key, None)
    parsed["effective_dtype"] = dtype
    parsed["gradient_checkpointing"] = checkpointing
    return parsed


def compare_config(reference, actual):
    left, right = config_parameters(reference), config_parameters(actual)
    missing = {"missing": True}
    differences = {
        k: {"expected": left.get(k, missing), "actual": right.get(k, missing)}
        for k in sorted(set(left) | set(right))
        if left.get(k, missing) != right.get(k, missing)
    }
    def canonical(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode("utf8")).hexdigest()
    return differences, canonical(left), canonical(right)
