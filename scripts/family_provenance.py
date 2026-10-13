"""Explicit provenance path mapping; never rewrite upstream hashes/configuration."""
from pathlib import Path
import json,hashlib

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resolve(name,mapping):
 if name in mapping.get('files',{}):return Path(mapping['files'][name])
 candidates=[]
 for item in mapping.get('prefixes',[]):
  remote=item['remote'].rstrip('/\\');normalized=name.replace('\\','/');remote=remote.replace('\\','/')
  if normalized==remote or normalized.startswith(remote+'/'):
   relative=normalized[len(remote):].lstrip('/');assert '..' not in relative.split('/');candidates.append((len(remote),Path(item['local'])/relative))
 if candidates:return max(candidates,key=lambda x:x[0])[1]
 p=Path(name)
 if p.exists():return p
 raise FileNotFoundError('No explicit local mapping for '+name)

def verify_manifest(path,path_map=None,include_config=False):
 mapping=json.loads(path_map.read_text()) if path_map else {};entries=json.loads(path.read_text());checks=[]
 def check(name,expected,kind):
  p=resolve(name,mapping);actual=sha(p);assert actual==expected,('hash mismatch',name,str(p),expected,actual);checks.append({'original_path':name,'local_path':str(p.resolve()),'sha256':actual,'kind':kind})
 for name,entry in entries.items():check(name,entry['sha256'] if isinstance(entry,dict) else entry,'manifest')
 if include_config:
  config=json.loads((path.parent/'config.json').read_text())
  for name,expected in config['input_sha256'].items():check(name,expected,'baseline config input')
  assert any(c['sha256']==config['script_sha256'] for c in checks),'Historical executed source snapshot not supplied/matched'
 return {'manifest_path':str(path),'manifest_sha256':sha(path),'path_map_sha256':sha(path_map) if path_map else None,'checks':checks,'upstream_files_modified':False}

def finalize(output,sources):
 paths=[p for p in output.iterdir() if p.is_file() and p.name!='manifest.json']+list(sources)
 (output/'manifest.json').write_text(json.dumps({str(p):sha(p) for p in paths},indent=2),encoding='utf-8')
