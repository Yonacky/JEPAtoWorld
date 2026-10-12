"""Archive new-route evidence after experiment completion; no remote operations."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def main():
 folders=['runs/spatial_information_retention_20261012','runs/nonlinear_information_20261013','runs/intervention_20261012/dynamics_audit_20261014','runs/intervention_20261012/dynamics_production_20261015','runs/causalworld_dynamics_readout_20261015']
 files=[]
 for name in folders:
  p=ROOT/name
  assert p.is_dir(),name
  files.extend(f for f in p.rglob('*') if f.is_file())
 for name in ['spatial_information_retention_20261012.py','nonlinear_information_20261013.py','audit_causalworld_dynamics_20261014.py','generate_causalworld_dynamics_20261015.py','complex_spatial_features.py','complex_route_comparison_20261012.py','finalize_other_routes_20261012.py']:
  files.append(ROOT/'scripts'/name)
 files.extend(f for f in (ROOT/'scripts').glob('*dynamics*20261015.py'))
 files.extend(f for f in (ROOT/'docs').glob('*dynamics*20261015.txt'))
 for name in ['spatial_information_retention_report_20261012.txt','nonlinear_information_report_20261013.txt','other_routes_report_20261012.txt','MASTER_PLAN.md']:
  files.append(ROOT/'docs'/name)
 files=sorted(set(files)); manifest={str(f.relative_to(ROOT)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files}
 out=ROOT/'experiments/other_routes_batch_20261012.zip'; assert not out.exists()
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in files:z.write(f,str(f.relative_to(ROOT)))
  z.writestr('archive_manifest.json',json.dumps(manifest,indent=2))
 with zipfile.ZipFile(out) as z:
  for rel,meta in manifest.items():assert hashlib.sha256(z.read(rel.replace(chr(92),'/'))).hexdigest()==meta['sha256'],rel
 record={'files':len(files),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'archive':str(out)}
 (ROOT/'runs/other_routes_backup_20261012.json').write_text(json.dumps(record,indent=2));print(record)
if __name__=='__main__':main()
