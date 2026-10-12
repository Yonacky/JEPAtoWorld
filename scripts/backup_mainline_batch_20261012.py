from pathlib import Path
import json,hashlib,zipfile
r=Path('F:/2610/CVPR27');out=r/'experiments/mainline_batch_20261012.zip';assert not out.exists()
files=set()
for n in ['mainline_history_20261012','mainline_fourway_20261012','mainline_regroup_20261012','intervention_20261012/edit_objective']:
    files.update(p for p in (r/'runs'/n).rglob('*') if p.is_file() and p.suffix!='.zip')
for n in ['mainline_history_20261012.py','mainline_fourway_20261012.py','mainline_regroup_20261012.py','round2_a_global.py','round2_b_shared_private.py','finalize_mainline_batch_20261012.py']:
    files.add(r/'scripts'/n)
for n in ['mainline_history_report_20261012.txt','mainline_fourway_report_20261012.txt','mainline_regroup_report_20261012.txt','history_boundary_audit_20261012.txt','edit_objective_report_20261012.txt','mainline_batch_report_20261012.txt','MASTER_PLAN.md']:
    files.add(r/'docs'/n)
files.update([r/'AGENTS.md',r/'Agent.md',r/'runs/mainline_batch_validation_20261012.json'])
for n in ['diagnose_edit_objective.py','report_edit_objective.py']:files.add(r/'runs/intervention_20261012/workspace/scripts'/n)
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED) as z:
    for p in sorted(files):assert p.exists(),str(p);z.write(p,str(p.relative_to(r)))
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    for n in z.namelist():assert hashlib.sha256(z.read(n)).digest()==hashlib.sha256((r/n).read_bytes()).digest(),n
manifest={'status':'PASS','files':len(files),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'archive':str(out.relative_to(r))}
(r/'runs/mainline_batch_backup_20261012.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
