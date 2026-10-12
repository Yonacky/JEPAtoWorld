from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
import numpy as np
from scipy import linalg as la
import json
r=Path('F:/2610/CVPR27');out=[]
for ds in ['pong','pinball']:
 z=np.load(r/f'runs/e04_inputs/cache/{ds}_ijepa_r0.npz');x=z['train'];t=np.load(r/f'runs/mainline_predictive_history_20261012/{ds}_truth_indices.npz')['rows_train'];m=np.load(r/f'runs/mainline_predictive_history_20261012/{ds}_rrr128_lambda0.01_transform.npz')
 h=np.concatenate([x[t-j] for j in range(4)],1)/2-m['history_center'];f=np.concatenate([x[t+j] for j in [1,2,4]],1)-m['future_center'];g=h.T@h/len(h)+.01*np.eye(512);c=h.T@f/len(h)
 L=la.cholesky(g,lower=True);Q=la.solve_triangular(L,c,lower=True);u,s,vt=la.svd(Q,full_matrices=False);w=la.solve_triangular(L.T,(u[:,:128]*s[:128])@vt[:128],lower=False);saved=m['B']@m['future_basis'].T
 error=float(np.max(np.abs(w-saved)));assert error<1e-7,error
 out.append({'dataset':ds,'alternative_cholesky_svd_weight_max_error':error})
(r/'runs/predictive_history_root_audit_20261012.json').write_text(json.dumps({'status':'PASS','checks':out},indent=2));print(out)
