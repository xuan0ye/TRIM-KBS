from pathlib import Path
import ast,json,hashlib,zipfile
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={}
for path in P.iterdir():
 if path.is_file()and(path.suffix in ['.py','.sh','.md']or path.name=='protocol.json'):
  if path.suffix=='.sh':path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
  if path.suffix=='.py':ast.parse(path.read_text('utf8'),filename=path.name)
  files[path.relative_to(P).as_posix()]=sha(path)
for folder in ['vendor','provenance']:
 for path in (P/folder).rglob('*'):
  if path.is_file()and path.suffix in ['.py','.json']:
   if path.suffix=='.py':ast.parse(path.read_text('utf8'),filename=str(path))
   files[path.relative_to(P).as_posix()]=sha(path)
(P/'bundle_manifest.json').write_text(json.dumps({'version':1,'files':files},indent=2),encoding='utf8')
archive=P.parent/(P.name+'.zip');package=dict(files)
package['bundle_manifest.json']=sha(P/'bundle_manifest.json')
for path in (P/'checks').glob('*.json'):package[path.relative_to(P).as_posix()]=sha(path)
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED)as z:
 for name in package:z.write(P/name,P.name+'/'+name)
 z.writestr(P.name+'/package_manifest.json',json.dumps(package,indent=2))
with zipfile.ZipFile(archive)as z:
 assert z.testzip()is None
 for name,want in package.items():assert hashlib.sha256(z.read(P.name+'/'+name)).hexdigest()==want
report=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),source_files=len(files),planned_training_jobs=75,formal_training_completed_locally=0)
(P/'delivery.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))