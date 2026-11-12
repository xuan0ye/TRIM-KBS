"""Pinned native-data identities and local BERT configuration checks."""
from pathlib import Path
import json
from diag_common import HERE,sha_file

def data_path(root,dataset,method):
 name='unaligned_39.pkl' if dataset=='SIMS' else ('unaligned_50.pkl' if method=='LNLN' else 'aligned_50.pkl')
 return root/'MSA Datasets'/dataset/'Processed'/name

def verify_data(root,dataset,method):
 p=data_path(root,dataset,method);rel=p.relative_to(root).as_posix()
 if method=='LNLN':
  expected={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (HERE/'native_data.sha1').read_text('utf8').splitlines() if line.strip()}
  algorithm='sha1';want=expected[rel]
 else:
  expected=json.loads((HERE/'trim_input_manifest.json').read_text('utf8'))['inputs']
  algorithm='sha256';want=next(r['sha256'] for r in expected if r['relative_path']==rel)
 got=sha_file(p,algorithm)
 if got!=want:raise RuntimeError(f'Processed data identity differs: {p}; {algorithm}={got}, expected {want}')
 return dict(path=str(p),algorithm=algorithm,digest=got)

def verify_bert(root,dataset,method):
 if method=='MulT':return None
 lang='bert_cn' if dataset=='SIMS' else 'bert_en';p=root/'MSA Datasets/pretrained_berts'/lang
 expected=json.loads((HERE/'bert_config_reference.json').read_text('utf8'))[lang]
 from transformers import BertConfig
 got=BertConfig.from_dict(json.loads((p/'config.json').read_text('utf8'))).to_dict()
 differences={k:{'expected':v,'observed':got.get(k)} for k,v in expected.items() if got.get(k)!=v}
 if differences:raise RuntimeError('BERT inference configuration differs: '+json.dumps(differences))
 if method=='LNLN' and not (p/'vocab.txt').is_file():raise FileNotFoundError(p/'vocab.txt')
 return dict(path=str(p),config_sha256=sha_file(p/'config.json'),semantic_config_equal=True,
  comparison='canonical architecture and forward-configuration keys; cosmetic metadata excluded; FP32 runtime')