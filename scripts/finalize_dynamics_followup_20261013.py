"""Archive new-route evidence after experiment completion; no remote operations."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def main():
 folders=['runs/causalworld_temporal_difference_20261013','runs/intervention_20261012/crossed_production_20261013','runs/causalworld_crossed_readout_20261013','runs/crossed_response_geometry_20261013']
 files=[]
 for name in folders:
  p=ROOT/name
  assert p.is_dir(),name
  files.extend(f for f in p.rglob('*') if f.is_file())
 files.extend(f for f in (ROOT/'scripts').glob('*crossed*20261013.py'))
 files.extend(f for f in (ROOT/'scripts').glob('*temporal_difference*20261013.py'))
 files.extend(f for f in (ROOT/'docs').glob('*crossed*20261013.txt'))
 files.extend(f for f in (ROOT/'docs').glob('*temporal_difference*20261013.txt'))
 for name in ['complex_spatial_features.py','evaluate_causalworld_dynamics_20261015.py','finalize_dynamics_followup_20261013.py']:
  files.append(ROOT/'scripts'/name)
 for name in ['dynamics_followup_report_20261013.txt','MASTER_PLAN.md']:
  files.append(ROOT/'docs'/name)
 files=sorted(set(files)); manifest={str(f.relative_to(ROOT)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files}
 out=ROOT/'experiments/dynamics_followup_batch_20261013.zip'; assert not out.exists()
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in files:z.write(f,str(f.relative_to(ROOT)))
  z.writestr('archive_manifest.json',json.dumps(manifest,indent=2))
 with zipfile.ZipFile(out) as z:
  for rel,meta in manifest.items():assert hashlib.sha256(z.read(rel.replace(chr(92),'/'))).hexdigest()==meta['sha256'],rel
 record={'files':len(files),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'archive':str(out)}
 (ROOT/'runs/dynamics_followup_backup_20261013.json').write_text(json.dumps(record,indent=2));print(record)
if __name__=='__main__':main()
