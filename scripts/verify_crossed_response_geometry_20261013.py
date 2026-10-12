from pathlib import Path
import numpy as np,json,hashlib
from scipy.linalg import helmert
r=Path(__file__).resolve().parents[1];o=r/'runs/crossed_response_geometry_20261013';data=json.loads((o/'results.json').read_text());z=np.load(o/'response_maps.npz');h=helmert(3).T;design=np.array([[1,*h[m],*h[f]] for m in range(3) for f in range(3)]);errs=[]
for row in data['results']:
 src,t=row['source'],row['step'];Y=z[f'{src}_{t}_responses'];co=np.linalg.lstsq(design,Y.reshape(40,9,-1).transpose(1,0,2).reshape(9,-1),rcond=None)[0];pred=(design@co).reshape(9,40,-1).transpose(1,0,2).reshape(Y.shape);res=Y-pred
 for sc in range(40):
  den=np.sum(Y[sc]**2);ratio=np.sum(res[sc]**2)/den if den>1e-20 else None;expected=row['scene_energy_decomposition'][sc]['interaction_fraction']
  if ratio is not None:errs.append(abs(ratio-expected));assert abs(ratio-expected)<1e-10
  else:assert expected is None
 for factor,R in [('mass',Y.mean(2)),('force',Y.mean(1))]:
  for q in [x for x in row['geometry'] if x['factor']==factor]:
   b=q['budget'];B=z[f'{src}_{t}_{factor}_rank{b}'];assert np.max(abs(B.T@B-np.eye(B.shape[1])),initial=0)<1e-10
   dv=R[32:].reshape(-1,R.shape[-1]);den=np.sum(dv**2);capt=np.sum((dv@B)**2)/den if den>1e-20 else None
   if capt is not None:assert abs(capt-q['dev_energy_capture'])<1e-10
   else:assert q['dev_energy_capture'] is None
   tr=R[:24].reshape(-1,R.shape[-1]);sv=z[f'{src}_{t}_{factor}_singular'];expected=B*sv[:B.shape[1]]**2;err=np.max(abs(tr.T@(tr@B)-expected),initial=0)/max(1,np.max(abs(expected),initial=0));assert err<1e-8
out={'status':'PASS','interaction_lstsq_max_error':max(errs),'checks':'Independent 9-cell linear design residual; saved basis orthogonality/eigen equations and dev projections; t0 NA'};(o/'independent_validation.json').write_text(json.dumps(out,indent=2));print(out)
