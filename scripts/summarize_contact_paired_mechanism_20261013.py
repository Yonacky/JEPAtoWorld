from pathlib import Path
import numpy as np,json,hashlib
B=Path('F:/2610/CVPR27');O=B/'runs/contact_paired_mechanism_20261013';r=json.loads((O/'results.json').read_text());rows=[];rng=np.random.default_rng(20261018)
for scope,scenes in [('all',list(range(40))),('dev',list(range(32,40)))]:
 ix=rng.integers(len(scenes),size=(5000,len(scenes)))
 for metric in ['off_xy_path_m','on_accx_mean','early5_accx_mean']:
  d=[]
  for s in scenes:
   q=[x for x in r['branches'] if x['scene']==s];d.append(np.mean([x[metric] for x in q if x['mode']=='off'])-np.mean([x[metric] for x in q if x['mode']=='on']))
  d=np.array(d);rows.append({'scope':scope,'metric':metric,'direction':'OFF minus ON, branch mean within scene','mean':float(d.mean()),'scene_ci95':np.percentile(d[ix].mean(1),[2.5,97.5]).tolist(),'per_scene':dict(zip(map(str,scenes),d.tolist()))})
(O/'paired_phase_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf8');print(json.dumps([{k:v for k,v in x.items() if k!='per_scene'} for x in rows],indent=2))
