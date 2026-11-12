"""Immutable source-only release of the expert pilot."""
from pathlib import Path
import argparse,hashlib,json,zipfile
HERE=Path(__file__).resolve().parent
PIN='77b9fd9e6e46cdcdbf7a3dcc4ca5471e9cd226e9e32f61a1f98f895258a70d43'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
 files=['expert_core.py','run_expert.py','remote_audio_expert.sh','PROTOCOL.md','REMOTE_README.md','build_bundle.py','checks/check_expert.py','checks/check_diagnosis.py']
 evidence=HERE/'checks/engineering_check.json'
 if not args.preflight:
  if not evidence.exists():raise RuntimeError('Run expert engineering checks first')
  check=json.loads(evidence.read_text(encoding='utf8'))
  if check['status']!='pass':raise RuntimeError('Engineering checks failed')
  for rel,digest in check['tested_source_sha256'].items():
   if sha(HERE/rel)!=digest:raise RuntimeError('Source changed since engineering check: '+rel)
 if evidence.exists():files.append('checks/engineering_check.json')
 smoke=HERE/'checks/diagnosis_smoke.json'
 if not args.preflight:
  report=json.loads(smoke.read_text(encoding='utf8'))
  if report['status']!='pass' or report['tested_core_sha256']!=sha(HERE/'expert_core.py'):raise RuntimeError('Missing matching complete-diagnosis smoke check')
 if smoke.exists():files.append('checks/diagnosis_smoke.json')
 if sha(HERE.parent/'paper_audio_supervised_fusion_20261028/bundle_manifest.json')!=PIN:raise RuntimeError('Prior fusion release changed')
 manifest=dict(release='audio_expert_mechanism_20261030',fusion_manifest_sha256=PIN,files={f:sha(HERE/f) for f in files},test_evaluated=False)
 (HERE/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
 archive=HERE.parent/'paper_audio_expert_20261030.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for f in files+['bundle_manifest.json']:z.write(HERE/f,HERE.name+'/'+f)
 print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),manifest_sha256=sha(HERE/'bundle_manifest.json'))))
if __name__=='__main__':main()
