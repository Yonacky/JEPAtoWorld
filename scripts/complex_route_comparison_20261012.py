"""G2 fixed complex-domain route comparison, episode-disjoint reused development."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json,hashlib,argparse,warnings,sys
import numpy as np
from scipy import linalg as la
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
AL=np.array([1e-6,1e-4,.01,.1,1,10,100])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def ew(e):return np.array([1/np.sum(e==v) for v in e])/len(np.unique(e))
def load(root,panel):
 if panel=='libero_mean':
  folder=root/'runs/complex_pilot_20261011/libero';fp=folder/'features.npz';mp=folder/'sample_manifest.npz';tp=folder/'transforms.npz';t=np.load(tp);mu=t['pca_mean'];P=t['pca_whitening'];Qs={'sfa':t['sfa'],'block4':t['full_block4']};stride=4;pool='mean';groups={'ee_position':[0,1,2],'gripper_joints':[3,4]}
 else:
  folder=root/'runs/spatial_holdout_20261011';fp=folder/'features/features.npz';mp=folder/'features/sample_manifest.npz';pool=panel.split('_')[-1];tp=folder/'evaluation/transforms_and_weights.npz';t=np.load(tp);mu=t[pool+'_mean'];P=t[pool+'64_pca'];Qs={};stride=16;groups={'object_position':[0,1,2],'object_green':[3],'velocity_proxy':[4,5,6],'floor_representative':[7],'stage_representative':[10]}
  if pool=='mean':
   tp=root/'runs/complex_pilot_20261011/causalworld/transforms.npz';t=np.load(tp);mu=t['pca_mean'];P=t['pca_whitening'];Qs={'sfa':t['sfa'],'block4':t['full_block4']}
 raw=np.load(fp)[pool].astype(float);m=dict(np.load(mp));masks=[m['split']==i for i in range(3)];X=[raw[b] for b in masks];W=[(x-mu)@P for x in X];Y=[m['labels'][b].astype(float) for b in masks];E=[m['episode'][b] for b in masks];F=[m['frame'][b] for b in masks];wt=[ew(e) for e in E]
 assert all(not set(E[i])&set(E[j]) for i in range(3) for j in range(i));assert all(len(set(zip(e.tolist(),f.tolist())))==len(e) for e,f in zip(E,F));assert np.max(np.abs(np.cov(W[0],rowvar=False)-np.eye(64)))<1e-5
 pair=np.flatnonzero((E[0][1:]==E[0][:-1])&(np.diff(F[0])==stride));assert len(pair)>64
 audit={'panel':panel,'source_files':{str(p):sha(p) for p in [fp,mp,tp]},'split_rows':[len(x) for x in X],'episodes':[np.unique(e).tolist() for e in E],'pair_count':len(pair),'stride':stride,'groups':groups,'labels':m['target_names'].tolist(),'scope':'Previously inspected dev; not new confirmation. No within-sequence reset guarantees; visual readout only.'}
 return X,W,Y,E,F,wt,m,P,mu,Qs,pair,groups,audit

def choose(x,y,w,k,block):
 a=x-np.sum(x*w[:,None],0);b=y-np.sum(y*w[:,None],0)
 if block:
  chunks=np.arange(x.shape[1]).reshape(-1,k);scores=[]
  for ids in chunks:
   z=a[:,ids]*np.sqrt(w[:,None]);fit=z@la.lstsq(z,b*np.sqrt(w[:,None]))[0];scores.append(np.sum(fit**2))
  return chunks[np.argmax(scores)]
 cov=(a*w[:,None]).T@b;score=np.sum(cov**2,1)/np.maximum(np.sum(a*a*w[:,None],0),1e-20);return np.argsort(-score,kind='stable')[:k]
def decomp(A,w):
 mx=np.sum(A[0]*w[:,None],0);u,s,v=la.svd((A[0]-mx)*np.sqrt(w[:,None]),full_matrices=False);return mx,u,s,v
def fit(A,Y,wt,dec=None):
 mx,u,s,v=decomp(A,wt[0]) if dec is None else dec;my=np.sum(Y[0]*wt[0][:,None],0);rhs=u.T@((Y[0]-my)*np.sqrt(wt[0][:,None]));coefs=[v.T@((s/(s*s+al))[:,None]*rhs) for al in AL];loss=[float(np.sum(wt[1]*np.mean(((A[1]-mx)@c+my-Y[1])**2,1))) for c in coefs];i=int(np.argmin(loss));return (A[2]-mx)@coefs[i]+my,coefs[i],mx,my,float(AL[i]),loss[i]
def run(root,O):
 O.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root/'core/e03'));from e03_core import optimize_joint
 import torch;torch.set_num_threads(2)
 cfg={'panels':['libero_mean','causalworld_mean','causalworld_grid'],'dim':64,'methods':['white','sfa','block4','ica0','ica1','ica2'],'ica':'parallel logcosh, prewhitened,1000 iterations,tol1e-4; no retry or dev choice','block':'Reuse verified original mean basis; grid exact existing optimize_joint,150 steps,train objective initializer selection','diagnostic':'Targets grouped, shared selected coordinates and alpha; train/tune episode-equal. top1/4/8 marginal OLS score; block4/8 joint OLS16/8 candidates. Different search families not equal search budgets.','references':'Raw full and privileged cubic progress are not compact information-equivalent methods.','fit_weights':'Transforms use original frame weighting; all diagnostic fitting/selection and primary metrics episode equal.','alphas':AL.tolist(),'scope':'CPU, reused complex dev exploration, not independent confirmation','code_sha256':sha(Path(__file__)),'core_sha256':sha(root/'core/e03/e03_core.py')};dump(O/'config.json',cfg);rows=[];audits=[];fits=[];inv=[];count=0
 for panel in cfg['panels']:
  X,W,Y,E,F,wt,m,P,mu,Qs,pair,groups,audit=load(root,panel);Qs={'white':np.eye(64),**Qs};ym=np.sum(Y[0]*wt[0][:,None],0);sd=np.sqrt(np.sum((Y[0]-ym)**2*wt[0][:,None],0));assert np.all(sd>1e-9);YS=[(y-ym)/sd for y in Y]
  if 'sfa' not in Qs:
   delta=W[0][pair+1]-W[0][pair];_,S=la.eigh(delta.T@delta/len(delta));Qs['sfa']=S;C=W[0][pair].T@W[0][pair+1]/len(pair);R=la.qr(np.random.default_rng(20261011).normal(size=(64,64)))[0];B,bf=optimize_joint(C[None],4,[np.eye(64),S,R],steps=150,device='cpu');Qs['block4']=B;fits.append({'panel':panel,'method':'block4','fit':bf})
  for seed in range(3):
   model=FastICA(whiten=False,algorithm='parallel',fun='logcosh',max_iter=1000,tol=1e-4,random_state=seed)
   with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always');model.fit(W[0])
   Qs[f'ica{seed}']=model.components_.T;fits.append({'panel':panel,'method':f'ica{seed}','iterations':int(model.n_iter_),'converged':not any(issubclass(w.category,ConvergenceWarning) for w in ws),'warnings':[str(w.message) for w in ws],'orth_error':float(np.max(np.abs(model.components_@model.components_.T-np.eye(64))))})
  np.savez_compressed(O/f'{panel}_maps.npz',pca=P,mean=mu,target_mean=ym,target_sd=sd,**Qs);full={};decs={}
  def evaluate(method,mode,Z,g,coords):
   nonlocal count
   ids=groups[g];A=[z[:,coords] for z in Z];yy=[y[:,ids] for y in YS]
   if mode=='full' and method not in decs:decs[method]=decomp(A,wt[0])
   p,c,mx,my,alpha,tl=fit(A,yy,wt,decs.get(method) if mode=='full' else None);p=p*sd[ids]+ym[ids];truth=Y[2][:,ids];eps=[]
   for e in np.unique(E[2]):
    b=E[2]==e;err=np.mean(((p[b]-truth[b])/sd[ids])**2);den=np.sum((truth[b]-truth[b].mean(0))**2);rr=None if den<1e-12 else float(1-np.sum((p[b]-truth[b])**2)/den);eps.append({'episode':str(e),'standardized_mse':float(err),'vector_rmse':float(np.sqrt(np.mean(np.sum((p[b]-truth[b])**2,1)))),'group_r2':rr})
   name=f'pred_{count:04d}.npz';count+=1;np.savez_compressed(O/name,prediction=p,truth=truth,episode=E[2],frame=F[2],coordinates=coords,coefficients=c,input_mean=mx,output_mean=my,target_indices=np.array(ids),alpha=alpha)
   den=np.sum((truth-truth.mean(0))**2);rows.append({'panel':panel,'method':method,'mode':mode,'group':g,'dimensions':len(coords),'macro_standardized_mse':float(np.mean([e['standardized_mse'] for e in eps])),'pooled_group_r2':None if den<1e-12 else float(1-np.sum((p-truth)**2)/den),'tune_macro_mse':tl,'alpha':alpha,'per_episode':eps,'file':name})
   if mode=='full' and method in Qs:full[method,g]=p
  for method,Q in Qs.items():
   assert np.max(np.abs(Q.T@Q-np.eye(64)))<1e-6;Z=[w@Q for w in W]
   for g,ids in groups.items():
    for k in [1,4,8]:evaluate(method,f'top{k}',Z,g,choose(Z[0],YS[0][:,ids],wt[0],k,False))
    for k in [4,8]:evaluate(method,f'block{k}',Z,g,choose(Z[0],YS[0][:,ids],wt[0],k,True))
    evaluate(method,'full',Z,g,np.arange(64))
   print(panel,method,'done',flush=True)
  pr=(m['frame']/np.maximum(m['episode_length']-1,1))[:,None];prog=np.concatenate([pr**k for k in [1,2,3]],1);ZZ=[prog[m['split']==s] for s in range(3)]
  for g in groups:evaluate('raw','full',X,g,np.arange(X[0].shape[1]));evaluate('progress_cubic','full',ZZ,g,np.arange(3))
  for method in Qs:
   for g in groups:
    err=float(np.max(np.abs(full[method,g]-full['white',g])));assert err<1e-5;inv.append({'panel':panel,'method':method,'group':g,'max_full_prediction_difference':err})
  audits.append(audit);dump(O/'results.json',rows);dump(O/'audits.json',audits);dump(O/'method_fit.json',fits);dump(O/'invariants.json',inv)
 dump(O/'COMPUTATION_COMPLETE.json',{'files':count,'status':'computed, independent verification pending'})
def verify(root,O):
 rows=json.loads((O/'results.json').read_text());err=0.;merr=0.;count=0
 for panel in ['libero_mean','causalworld_mean','causalworld_grid']:
  X,W,Y,E,F,wt,m,P,mu,Qs,pair,groups,audit=load(root,panel);maps=np.load(O/f'{panel}_maps.npz');ym=maps['target_mean'];sd=maps['target_sd'];YS=[(y-ym)/sd for y in Y];pr=(m['frame']/np.maximum(m['episode_length']-1,1))[:,None];pr=np.concatenate([pr**k for k in [1,2,3]],1)
  for row in [r for r in rows if r['panel']==panel]:
   z=np.load(O/row['file']);ids=z['target_indices'];coords=z['coordinates'];method=row['method'];Z=X if method=='raw' else ([pr[m['split']==s] for s in range(3)] if method=='progress_cubic' else [w@maps[method] for w in W]);A=[a[:,coords] for a in Z];mx=np.sum(A[0]*wt[0][:,None],0);my=np.sum(YS[0][:,ids]*wt[0][:,None],0);xx=(A[0]-mx)*np.sqrt(wt[0][:,None]);yy=(YS[0][:,ids]-my)*np.sqrt(wt[0][:,None]);alpha=float(z['alpha']);d=xx.shape[1]
   c=xx.T@la.solve(xx@xx.T+alpha*np.eye(len(xx)),yy,assume_a='pos') if d>len(xx) else la.solve(xx.T@xx+alpha*np.eye(d),xx.T@yy,assume_a='pos');p=((A[2]-mx)@c+my)*sd[ids]+ym[ids];err=max(err,float(np.max(np.abs(p-z['prediction']))));assert np.array_equal(z['truth'],Y[2][:,ids]) and np.array_equal(z['episode'],E[2]) and np.array_equal(z['frame'],F[2]);score=np.sum(np.mean(((p-Y[2][:,ids])/sd[ids])**2,1)*wt[2]);merr=max(merr,abs(float(score)-row['macro_standardized_mse']));count+=1
 assert err<1e-5 and merr<1e-6
 result={'status':'passed','files':count,'prediction_max_error':err,'macro_mse_max_error':merr,'episode_frame_targets':'exact source match','recompute':'Direct primal/dual normal equations, independent of main thin SVD','code_sha256':sha(Path(__file__))};dump(O/'independent_validation.json',result);dump(O/'COMPLETE.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args();(verify if a.verify else run)(a.root,a.output)
