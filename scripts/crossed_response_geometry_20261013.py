"""Predeclared two-factor response decomposition; labeled intervention diagnostic."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import numpy as np
from scipy.linalg import svd
import json,hashlib,argparse

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 z=np.load(a.data);f=np.load(a.features);fm=np.load(a.features.parent/'sample_manifest.npz')
 for k in ['split','episode','frame','labels']:assert np.array_equal(z[k],fm[k]),k
 sources={'grid':f['grid'].astype(float),'xyz':z['actual_states'][:,0,:3]};rows=[];saved={};errors=[]
 for source,A in sources.items():
  for t in [0,30,120]:
   Y=[]
   for sc in range(40):
    cells=[]
    for m in range(3):
     for force in range(3):
      ix=np.flatnonzero((z['scene']==sc)&(z['mass_level']==m)&(z['force_level']==force)&(z['frame']==t));assert len(ix)==1;cells.append(A[ix[0]])
    Y.append(np.array(cells).reshape(3,3,-1))
   Y=np.array(Y);D=Y-Y[:,:1,:1];grand=D.mean((1,2),keepdims=True);centered=D-grand;M=D.mean(2,keepdims=True)-grand;F=D.mean(1,keepdims=True)-grand;I=centered-M-F
   scale=max(1.,float(np.sum(centered**2)));orth=max(abs(float(np.sum(M*F))),abs(float(np.sum(M*I))),abs(float(np.sum(F*I))))/scale;assert orth<1e-10
   assert np.max(abs(centered-(M+F+I)))<1e-10
   total=(centered**2).sum((1,2,3));em=3*(M**2).sum((1,2,3));ef=3*(F**2).sum((1,2,3));ei=(I**2).sum((1,2,3));err=float(np.max(abs(total-em-ef-ei))/max(1.,float(total.max())));assert err<1e-10;errors.append(err)
   if t==0:assert np.max(abs(Y-Y[:,:1,:1]))==0
   per=[]
   for sc in range(40):per.append({'scene':sc,'total_energy':float(total[sc]),'mass_fraction':float(em[sc]/total[sc]) if total[sc]>1e-20 else None,'force_fraction':float(ef[sc]/total[sc]) if total[sc]>1e-20 else None,'interaction_fraction':float(ei[sc]/total[sc]) if total[sc]>1e-20 else None})
   geometry=[];U={}
   for name,R in [('mass',M.reshape(40,3,-1)),('force',F.reshape(40,3,-1))]:
    train=R[:24].reshape(-1,R.shape[-1]);_,s,v=svd(train,full_matrices=False);rank=int(np.sum(s>max(s[0]*1e-8,1e-12)));saved[f'{source}_{t}_{name}_singular']=s
    for budget in [1,2,4]:
     k=min(rank,budget);B=v[:k].T;U[name,budget]=B;saved[f'{source}_{t}_{name}_rank{budget}']=B;dev=R[32:];den=(dev**2).sum((1,2));num=((dev@B)**2).sum((1,2));assert np.all(num<=den+1e-9)
     geometry.append({'factor':name,'budget':budget,'actual_rank':k,'dev_energy_capture':float(num.sum()/den.sum()) if den.sum()>1e-20 else None,'dev_scene_capture':[float(n/d) if d>1e-20 else None for n,d in zip(num,den)]})
   overlaps=[]
   for budget in [1,2,4]:
    B,C=U['mass',budget],U['force',budget];d=min(B.shape[1],C.shape[1]);overlaps.append({'budget':budget,'normalized_projection_overlap':float(np.sum((B.T@C)**2)/d) if d else None})
   rows.append({'source':source,'step':t,'scene_energy_decomposition':per,'geometry':geometry,'overlap':overlaps});saved[f'{source}_{t}_responses']=centered
 out={'results':rows,'energy_identity_relative_max_error':max(errors),'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.data,a.features,Path(__file__)]},'scope':'Known intervention axes, raw Euclidean geometry, train-only subspaces, no uniqueness claim'}
 (a.output/'results.json').write_text(json.dumps(out,indent=2),encoding='utf-8');np.savez_compressed(a.output/'response_maps.npz',**saved);print('response geometry complete',max(errors))
if __name__=='__main__':main()
