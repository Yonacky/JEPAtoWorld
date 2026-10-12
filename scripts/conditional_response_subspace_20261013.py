"""G5g: labeled response-subspace transfer conditioned on unlabeled initial images."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.linalg import svd,eigh
SEEDS=[20261013,20261014,20261015]
METHODS=['global24','local8']+[f'random8_{s}' for s in SEEDS]+['oracle_dev']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def basis(X):
 u,s,v=svd(X,full_matrices=False);k=min(4,int(np.sum(s>max(s[0]*1e-8,1e-12))));return v[:k].T

def load(data,features):
 z=np.load(data);f=np.load(features);fm=np.load(features.parent/'sample_manifest.npz');v=json.loads((data.parent/'validation.json').read_text());assert v['status']=='PASS_INVARIANTS'
 for k in ['split','episode','frame','labels']:assert np.array_equal(z[k],fm[k])
 A=f['grid'].astype(float);assert A.shape==(3240,5120) and np.isfinite(A).all();indices=np.empty((40,3,3,3),int)
 for sc in range(40):
  ix=np.flatnonzero(z['scene']==sc);assert len(ix)==81 and np.all(z['split'][ix]==(0 if sc<24 else 1 if sc<32 else 2))
  for ti,t in enumerate([0,30,120]):
   for m in range(3):
    for force in range(3):
     ix=np.flatnonzero((z['scene']==sc)&(z['mass_level']==m)&(z['force_level']==force)&(z['frame']==t));assert len(ix)==1;indices[sc,ti,m,force]=ix[0]
  vals=A[indices[sc,0]];assert np.array_equal(vals,np.broadcast_to(vals[0,0],vals.shape))
 Y=A[indices];return Y,indices

def response(Y,independent=False):
 out={}
 for ti,t in enumerate([0,30,120]):
  D=Y[:,ti]-Y[:,ti,:1,:1];g=D.mean((1,2),keepdims=True)
  if independent:
   centered=Y[:,ti]-Y[:,ti].mean((1,2),keepdims=True);M=centered.sum(2)/3;F=centered.sum(1)/3
  else:M=(D.mean(2,keepdims=True)-g).reshape(40,3,-1);F=(D.mean(1,keepdims=True)-g).reshape(40,3,-1)
  out[t,'mass']=M;out[t,'force']=F
 return out

def training(R,method,sc,neighbors):
 if method=='global24':ids=np.arange(24)
 elif method=='oracle_dev':ids=np.array([sc])
 else:ids=neighbors[method][sc-32]
 return R[ids].reshape(-1,R.shape[-1]),ids

def run(a):
 O=a.output;O.mkdir(parents=True,exist_ok=False);Y,indices=load(a.data,a.features);R=response(Y);init=Y[:,0,0,0];mu=init[:24].mean(0);u,s,v=svd(init[:24]-mu,full_matrices=False);k=min(4,int(np.sum(s>max(s[0]*1e-8,1e-12))));P=v[:k].T;des=(init-mu)@P;dist=((des[32:,None]-des[None,:24])**2).sum(-1);neighbors={'local8':np.argsort(dist,axis=1,kind='stable')[:,:8]}
 for seed in SEEDS:
  rng=np.random.default_rng(seed);neighbors[f'random8_{seed}']=np.array([rng.choice(24,8,replace=False) for _ in range(8)])
 np.savez_compressed(O/'mapping.npz',source_indices=indices,descriptor_mean=mu,descriptor_basis=P,descriptor_singular=s,descriptors=des,squared_distances=dist,**neighbors)
 for name,r in R.items():
  if name[0]==0:assert np.count_nonzero(r)==0
 rows=[];saved={};overlaps=[]
 for t in [30,120]:
  for method in METHODS:
   for sc in range(32,40):
    bs={}
    for factor in ['mass','force']:
     X,ids=training(R[t,factor],method,sc,neighbors);B=basis(X);bs[factor]=B;held=R[t,factor][sc];pred=held@B@B.T;energy=float(np.sum(held**2));capture=float(np.sum((held@B)**2)/energy);key=f't{t}_{method}_s{sc}_{factor}';saved[key+'_basis']=B;saved[key+'_projected_response']=pred
     rows.append({'step':t,'method':method,'scene':sc,'factor':factor,'rank':B.shape[1],'training_scenes':ids.tolist(),'capture':capture,'response_energy':energy,'residual_fraction':float(np.sum((held-pred)**2)/energy),'file_key':key});assert abs(capture+rows[-1]['residual_fraction']-1)<1e-10
     if method=='oracle_dev':assert B.shape[1]<=2 and abs(capture-1)<1e-10
    B,C=bs['mass'],bs['force'];overlaps.append({'step':t,'method':method,'scene':sc,'mass_force_projection_overlap':float(np.sum((B.T@C)**2)/min(B.shape[1],C.shape[1]))})
 np.savez_compressed(O/'subspaces_and_projections.npz',**saved);summaries=[];paired=[]
 for t in [30,120]:
  for factor in ['mass','force']:
   for method in METHODS:
    vals=[r for r in rows if (r['step'],r['factor'],r['method'])==(t,factor,method)];acc=np.array([r['capture'] for r in vals]);rng=np.random.default_rng(571);ci=np.percentile(acc[rng.integers(8,size=(5000,8))].mean(1),[2.5,97.5]);summaries.append({'step':t,'factor':factor,'method':method,'scene_equal_capture':float(acc.mean()),'scene_ci95':ci.tolist(),'energy_weighted_capture':float(np.average(acc,weights=[r['response_energy'] for r in vals]))})
   local=np.array([r['capture'] for r in rows if (r['step'],r['factor'],r['method'])==(t,factor,'local8')])
   for other in ['global24']+[f'random8_{s}' for s in SEEDS]:
    prior=np.array([r['capture'] for r in rows if (r['step'],r['factor'],r['method'])==(t,factor,other)]);diff=local-prior;rng=np.random.default_rng(571);paired.append({'step':t,'factor':factor,'comparison':'local8 minus '+other,'mean_difference':float(diff.mean()),'paired_scene_ci95':np.percentile(diff[rng.integers(8,size=(5000,8))].mean(1),[2.5,97.5]).tolist()})
 for q in overlaps:
  vals=[]
  for factor in ['mass','force']:
   pre=f"t{q['step']}_";tail=f"_s{q['scene']}_{factor}_basis";B=saved[pre+q['method']+tail];C=saved[pre+'global24'+tail];vals.append(float(np.sum((B.T@C)**2)/min(B.shape[1],C.shape[1])))
  q['mass_and_force_overlap_with_global']=vals
 dump(O/'results.json',{'rows':rows,'summaries':summaries,'paired':paired,'overlaps':overlaps});dump(O/'config.json',{'seeds':SEEDS,'steps':[30,120],'descriptor':'train24 centered unwhitened PCA<=4; t0 identical across9 branches; stable Euclidean nearest8','response':'Known intervention marginal mass/force main effects; raw uncentered SVD rank<=4; no per-scene energy normalization','scope':'Previously seen dev exploratory; supervised response subspaces with unsupervised initial-image neighbor selector; oracle uses dev labels/responses and is not deployable','oracle':'Per-scene3 centered levels rank<=2; capture100% is algebraic','scene_split':[24,8,8],'tune':'Unused; no hyperparameter selection','inputs_sha256':{str(p):sha(p) for p in [a.data,a.features,a.features.parent/'sample_manifest.npz',Path(__file__)]}});print('computed',len(rows),flush=True)

def verify(a):
 O=a.output;Y,ix=load(a.data,a.features);R=response(Y,True);ref=response(Y);mp=np.load(O/'mapping.npz');z=np.load(O/'subspaces_and_projections.npz');res=json.loads((O/'results.json').read_text());assert np.array_equal(ix,mp['source_indices']);np.testing.assert_allclose((Y[:,0,0,0]-mp['descriptor_mean'])@mp['descriptor_basis'],mp['descriptors'],atol=1e-12);ds=((mp['descriptors'][32:,None]-mp['descriptors'][None,:24])**2).sum(-1);assert np.array_equal(np.argsort(ds,axis=1,kind='stable')[:,:8],mp['local8']);neighbors={k:mp[k] for k in ['local8']+[f'random8_{s}' for s in SEEDS]};maxerr=0.;maperr=0.
 for key in R:np.testing.assert_allclose(R[key],ref[key],atol=1e-12)
 for r in res['rows']:
  X,ids=training(ref[r['step'],r['factor']],r['method'],r['scene'],neighbors);assert ids.tolist()==r['training_scenes'];B=z[r['file_key']+'_basis'];vals,U=eigh(X@X.T);ord=np.argsort(vals)[::-1][:r['rank']];C=X.T@U[:,ord]/np.sqrt(vals[ord]);assert np.max(abs(C.T@C-np.eye(r['rank'])))<1e-7;held=R[r['step'],r['factor']][r['scene']];pred=held@C@C.T;err=float(np.max(abs(pred-z[r['file_key']+'_projected_response'])));maxerr=max(maxerr,err);assert err<1e-8;capture=float(np.sum(pred**2)/np.sum(held**2));maperr=max(maperr,abs(capture-r['capture']));assert abs(capture-r['capture'])<1e-8
 for seed in SEEDS:
  rng=np.random.default_rng(seed);assert np.array_equal(neighbors[f'random8_{seed}'],np.array([rng.choice(24,8,replace=False) for _ in range(8)]))
 for q in res['paired']:
  other=q['comparison'].split(' minus ')[1];get=lambda method:np.array([r['capture'] for r in res['rows'] if (r['step'],r['factor'],r['method'])==(q['step'],q['factor'],method)]);diff=get('local8')-get(other);rng=np.random.default_rng(571);np.testing.assert_array_equal(np.percentile(rng.choice(diff,(5000,8)).mean(1),[2.5,97.5]),q['paired_scene_ci95'])
 dump(O/'independent_validation.json',{'status':'PASS','maps':len(res['rows']),'max_projection_error':maxerr,'max_capture_error':maperr,'method':'Independent dual Gram eigendecomposition vs raw SVD; alternate marginal arithmetic; neighbors/source-index/bootstrap recomputed','code_sha256':sha(Path(__file__))});print('verified',maxerr,maperr)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args();(verify if a.verify else run)(a)
