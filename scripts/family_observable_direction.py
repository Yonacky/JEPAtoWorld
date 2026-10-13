"""G7e image-family fork of frozen G5k; no parameter changes."""
import contact_ratio_readout_20261013 as base
import family_provenance as provenance
from pathlib import Path
import argparse,json
import numpy as np
from scipy import linalg as la
SEED=20261013

def run(a):
 O=a.output;O.mkdir(parents=True,exist_ok=False);d=np.load(a.data);f=np.load(a.features);fm=np.load(a.features.parent/'sample_manifest.npz');A=f['grid'].astype(float)
 for k in ['split','episode','frame','labels']:np.testing.assert_array_equal(d[k],fm[k])
 audit=provenance.verify_manifest(a.baseline/'manifest.json',a.path_map,include_config=True)
 base.dump(O/'baseline_provenance_audit.json',audit)
 for required in [a.data,a.features,a.features.parent/'sample_manifest.npz']:
  assert any(Path(c['local_path']).resolve()==required.resolve() and c['sha256']==base.sha(required) for c in audit['checks']),('CLI input not in verified baseline provenance',str(required))
 cfg=json.loads((a.baseline/'config.json').read_text());assert cfg['input_kind']=='image_g5j','Only image_g5j enabled in this migration'
 assert A.ndim==2 and A.shape[0]==6480 and np.isfinite(A).all()
 rows=[];checks=[];mapping={};maxerr=0.
 for mode in ['on','off']:
  p=np.load(a.baseline/(mode+'_pca.npz'));b=np.load(a.baseline/(mode+'_logratio.npz'));idx=p['source_indices'];sc=p['scene'];sp=p['split'];assert idx.shape==(360,9)
  np.testing.assert_array_equal(d['frame'][idx],np.tile([0,1,5,10,20,30,60,90,120],(360,1)));assert np.all(d['contact_mode'][idx]==mode);assert np.all(d['scene'][idx]==sc[:,None]);assert np.all(d['split'][idx]==sp[:,None]);assert np.all(sp==np.where(sc<24,0,np.where(sc<32,1,2)))
  X=np.concatenate([A[idx[:,5]]-A[idx[:,0]],A[idx[:,8]]-A[idx[:,0]]],1);tr=sp==0;tu=sp==1;de=sp==2;P=p['basis'];assert P.shape==(2*A.shape[1],32);np.testing.assert_allclose(X[tr].mean(0),p['mean'],rtol=0,atol=1e-12);Z=(X-p['mean'])@P;y=np.log(d['labels'][idx[:,0],2]/d['labels'][idx[:,0],0]);mu=float(b['target_mean']);sd=float(b['target_std']);np.testing.assert_allclose([mu,sd],[y[tr].mean(),y[tr].std()],atol=1e-14);yt=(y[tr]-mu)/sd;w0=b['coefficients'];u=w0/la.norm(w0);N=la.null_space(u[None,:]);assert N.shape==(32,31);assert np.max(abs(N.T@u))<1e-12
  np.testing.assert_allclose(Z[de]@w0*sd+mu,b['prediction'],atol=1e-12);perm=np.arange(tr.sum());rng=np.random.default_rng(SEED)
  for s in range(24):
   loc=np.flatnonzero(sc[tr]==s);assert len(loc)==9;perm[loc]=rng.permutation(loc)
  mapping[mode+'_direction']=u;mapping[mode+'_residual_basis']=N;mapping[mode+'_permutation']=perm;mapping[mode+'_source_indices']=idx;mapping[mode+'_scene']=sc;mapping[mode+'_split']=sp;mapping[mode+'_pca_scores']=Z
  methods=[(f'pca{k}',np.eye(32)[:,:k],False) for k in [1,2,4,8,16,32]]+[('supervised1',u[:,None],False),('residual31',N,False),('shuffled32',np.eye(32),True)]
  for method,Q,shuffled in methods:
   zz=Z@Q;xx=[zz[t] for t in [tr,tu,de]];target=yt[perm] if shuffled else yt
   if method=='supervised1':
    coef=np.array([la.norm(w0)]);alpha=float(b['alpha']);loss=b['tune_losses'];pred=xx[2]@coef*sd+mu;ip=b['prediction'];rank=1
   else:
    G=xx[0].T@xx[0]/tr.sum();rhs=xx[0].T@target/tr.sum();ws=[la.solve(G+a0*np.eye(Q.shape[1]),rhs,assume_a='pos') for a0 in base.ALPHAS];loss=np.array([np.mean((xx[1]@w-(y[tu]-mu)/sd)**2) for w in ws]);best=int(loss.argmin());coef=ws[best];alpha=base.ALPHAS[best];pred=xx[2]@coef*sd+mu;rank=Q.shape[1]
    uu,ss,vv=la.svd(xx[0],full_matrices=False);iw=[vv.T@((ss/(ss*ss+tr.sum()*a0))*(uu.T@target)) for a0 in base.ALPHAS];il=np.array([np.mean((xx[1]@w-(y[tu]-mu)/sd)**2) for w in iw]);assert il.argmin()==best;ip=xx[2]@iw[best]*sd+mu
   err=float(np.max(abs(ip-pred)));assert err<1e-9;maxerr=max(maxerr,err);key=mode+'_'+method;np.savez_compressed(O/(key+'.npz'),prediction=pred,truth=y[de],scene=sc[de],source_indices=idx[de],coefficients=coef,pca32_projection=Q,alpha=alpha,tune_losses=loss,target_mean=mu,target_std=sd)
   metric=base.metrics(y[de],pred,sc[de],sd);mse=float(np.mean((y[de]-pred)**2));assert abs(mse-metric['log_mse'])<1e-12
   if method in ['supervised1','pca32']:np.testing.assert_allclose(pred,b['prediction'],rtol=0,atol=1e-11)
   if method=='residual31':np.testing.assert_allclose(Q@Q.T,np.eye(32)-np.outer(u,u),atol=1e-12)
   row={'mode':mode,'method':method,'dimension':rank,'alpha':alpha,**metric};e=np.array(list(metric['scene_log_mse'].values()));ref=np.array([np.mean((b['prediction'][sc[de]==s]-y[de][sc[de]==s])**2) for s in range(32,40)]);delta=e-ref;rng=np.random.default_rng(571);row['paired_mse_difference_vs_pca32']=float(delta.mean());row['paired_scene_ci95']=np.percentile(delta[rng.integers(8,size=(5000,8))].mean(1),[2.5,97.5]).tolist();rows.append(row);checks.append({'model':key,'svd_prediction_max_error':err});print(mode,method,'r2',metric['r2'],flush=True)
 np.testing.assert_array_equal(mapping['on_permutation'],mapping['off_permutation']);np.savez_compressed(O/'mapping.npz',**mapping);base.dump(O/'results.json',rows);base.dump(O/'independent_validation.json',{'status':'PASS','models':len(rows),'prediction_max_error':maxerr,'checks':checks,'scope':'Independent SVD readout, same saved PCA; PCA32 and supervised1 predictions equal historical G5j; residual projector identity checked'})
 base.dump(O/'config.json',{'ranks':[1,2,4,8,16,32],'alphas':base.ALPHAS,'shuffle_seed':SEED,'shuffle':'within train scene only; same permutation ON/OFF; tune/dev retain true labels; full32 only','input':'G5j grid concat(delta30,delta120), saved train-only PCA32 and original target scale','supervised1':'normalize historical ridge coefficients; reuse coefficient norm and original intercept, no refit; exact prediction identity, not a discovery','residual31':'Euclidean orthogonal complement of historical PCA32 ridge coefficient vector; refit original5alphas via true tune MSE','scope':'Previously inspected dev exploratory; known ratio label; no new information/capacity claim for supervised1','input_hashes':{str(p):base.sha(p) for p in [a.data,a.features,a.features.parent/'sample_manifest.npz',a.baseline/'manifest.json',Path(__file__),Path(base.__file__)]}})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--path-map',type=Path);a=p.parse_args();run(a);provenance.finalize(a.output,[Path(__file__),Path(base.__file__),Path(provenance.__file__)])
