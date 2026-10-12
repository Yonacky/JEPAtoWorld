"""T-3 pre-frozen new-task 12/30 replication. Task-local training, fixed ten-demo dev."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
from scipy import linalg as la
import complex_route_comparison_20261012 as common
common.AL=np.array([1e-7,1e-5,.001,.01,.1,1,10,100])
SEEDS=[20261012,20261013,20261014]
GROUPS={'position':[0,1,2],'gripper':[3,4]}
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ordered(x):return list(dict.fromkeys(x.tolist()))
def load(task,old=None):
 x=np.load(task/'features.npz')['mean'].astype(float);m=dict(np.load(task/'sample_manifest.npz'))
 assert x.shape[1]==1280 and np.isfinite(x).all()
 groups=[ordered(m['episode'][m['split']==i]) for i in range(3)]
 assert list(map(len,groups))==[30,10,10]
 assert all(not set(groups[i])&set(groups[j]) for i in range(3) for j in range(i))
 assert len(set(zip(m['episode'].tolist(),m['frame'].tolist())))==len(x)
 canonical=None
 for filename in ['train_order.json','data.json']:
  p=task/filename
  if p.exists():
   data=json.loads(p.read_text());order=data.get('training_order',data.get('train_order'))
   if order is not None:
    assert len(order)==30 and set(order)==set(groups[0])
    if canonical is not None:assert order==canonical
    canonical=order
 assert canonical is not None, 'Explicit training_order required'
 base12=canonical[:12];base=np.isin(m['episode'],base12);w=common.ew(m['episode'][base]);ym=w@m['labels'][base];sd=np.sqrt(w@((m['labels'][base]-ym)**2));assert np.all(sd>1e-9)
 return x,m,[base12,groups[1],groups[2]],[('n12',base12),('n30',canonical)],ym,sd

def run(root,task,old,O):
 O.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root/'core/e03'));from e03_core import optimize_joint
 import torch;torch.set_num_threads(2)
 x,m,oldgroups,plans,ym,sd=load(task,old);cfg={'task':str(task),'code_sha256':sha(Path(__file__)),'dependency_sha256':sha(Path(common.__file__)),'input_hashes':{str(p):sha(p) for p in [task/'features.npz',task/'sample_manifest.npz',task/'train_order.json',task/'data.json'] if p.exists()},'plans':{k:v for k,v in plans},'tune':oldgroups[1],'dev':oldgroups[2],'scale':'Frozen first12 train demo-equal mean and SD for both sizes','group_definitions':GROUPS,'methods':['white','sfa','full_block4'],'readout':'Shared coordinates/one alpha per group; train marginal OLS energy top8, all64; tune group normalized MSE; demo equal','alphas':common.AL.tolist(),'block':'150 steps, identity/SFA/random seed20261012 init chosen only train objective; same budget each size','split_seed':20261013,'scope':'Frozen-method new-task replication, independently fit each task; not zero-shot. First12 nested within30, fixed10tune/10dev. No result-based config changes.','device':'CPU'};dump(O/'config.json',cfg);rows=[];audits=[];invariants=[]
 for plan,demos in plans:
  masks=[np.isin(m['episode'],demos),m['split']==1,m['split']==2];X=[x[b] for b in masks];Y=[m['labels'][b].astype(float) for b in masks];E=[m['episode'][b] for b in masks];F=[m['frame'][b] for b in masks];wt=[common.ew(e) for e in E];YS=[(y-ym)/sd for y in Y];assert all(not set(E[i])&set(E[j]) for i in range(3) for j in range(i));mu=wt[0]@X[0];u,s,vh=la.svd((X[0]-mu)*np.sqrt(wt[0][:,None]),full_matrices=False);assert s[63]>s[0]*1e-8;P=vh[:64].T/s[:64];W=[(a-mu)@P for a in X];assert np.max(np.abs(W[0].T@(W[0]*wt[0][:,None])-np.eye(64)))<1e-8
  lookup={(str(e),int(f)):i for i,(e,f) in enumerate(zip(E[0],F[0]))};pair=np.array([(i,lookup[str(e),int(f)+4]) for i,(e,f) in enumerate(zip(E[0],F[0])) if (str(e),int(f)+4) in lookup]);assert len(pair)>64;pw=common.ew(E[0][pair[:,0]]);d=W[0][pair[:,1]]-W[0][pair[:,0]];sv,S=la.eigh(d.T@(d*pw[:,None]));C=W[0][pair[:,0]].T@(W[0][pair[:,1]]*pw[:,None]);R=la.qr(np.random.default_rng(20261012).normal(size=(64,64)))[0];B,bfit=optimize_joint(C[None],4,[np.eye(64),S,R],steps=150,device='cpu');Qs={'white':np.eye(64),'sfa':S,'full_block4':B};np.savez_compressed(O/f'{plan}_maps.npz',mean=mu,pca=P,sfa=S,full_block4=B,white=np.eye(64),target_mean=ym,target_sd=sd,train_rows=np.flatnonzero(masks[0]),tune_rows=np.flatnonzero(masks[1]),dev_rows=np.flatnonzero(masks[2]),pairs=pair)
  full={}
  for method,Q in Qs.items():
   A=[w@Q for w in W]
   for group,ids in GROUPS.items():
    for k in [8,64]:
     coords=common.choose(A[0],YS[0][:,ids],wt[0],k,False) if k<64 else np.arange(64);Z=[a[:,coords] for a in A];p,c,mx,my,al,tl=common.fit(Z,[y[:,ids] for y in YS],wt);p=p*sd[ids]+ym[ids];truth=Y[2][:,ids];epmetrics=[]
     for e in np.unique(E[2]):
      ix=E[2]==e;err=p[ix]-truth[ix];den=np.sum((truth[ix]-truth[ix].mean(0))**2);epmetrics.append({'episode':str(e),'normalized_mse':float(np.mean((err/sd[ids])**2)),'physical_mse':float(np.mean(err**2)),'vector_rmse':float(np.sqrt(np.mean(np.sum(err**2,axis=1)))),'r2':None if den<1e-14 else float(1-np.sum(err**2)/den)})
     name=f'{plan}_{method}_{group}_k{k}.npz';np.savez_compressed(O/name,prediction=p,truth=truth,episode=E[2],frame=F[2],coordinates=coords,target_indices=ids,coefficients=c,input_mean=mx,output_mean=my,alpha=al)
     rows.append({'plan':plan,'train_demos':len(demos),'method':method,'group':group,'budget':k,'normalized_mse':float(np.mean([e['normalized_mse'] for e in epmetrics])),'physical_mse':float(np.mean([e['physical_mse'] for e in epmetrics])),'per_episode':epmetrics,'alpha':al,'tune_normalized_mse':tl,'file':name})
     if k==64:full[method,group]=p
  for method in Qs:
   for g in GROUPS:
    err=float(np.max(np.abs(full[method,g]-full['white',g])));assert err<1e-7;invariants.append({'plan':plan,'method':method,'group':g,'full_prediction_difference':err})
  audits.append({'plan':plan,'demos':demos,'rows':[len(a) for a in X],'pair_count':len(pair),'pca_variance_retained':float(np.sum(s[:64]**2)/np.sum(s**2)),'block_fit':bfit,'pca_whiten_error':float(np.max(np.abs(W[0].T@(W[0]*wt[0][:,None])-np.eye(64))))});dump(O/'results.json',rows);dump(O/'audits.json',audits);dump(O/'invariants.json',invariants);print(plan,'complete',flush=True)
 dump(O/'COMPUTATION_COMPLETE.json',{'files':len(rows),'status':'computed; verification pending'})

def verify(root,task,old,O):
 x,m,oldgroups,plans,ym,sd=load(task,old);rows=json.loads((O/'results.json').read_text());err=0.;merr=0.;count=0
 for plan,demos in plans:
  z=np.load(O/f'{plan}_maps.npz');ix=[z[s+'_rows'] for s in ['train','tune','dev']];E=[m['episode'][i] for i in ix];wt=[common.ew(e) for e in E];X=[(x[i]-z['mean'])@z['pca'] for i in ix];Y=[(m['labels'][i]-ym)/sd for i in ix]
  for row in [r for r in rows if r['plan']==plan]:
   pz=np.load(O/row['file']);ids=pz['target_indices'];coords=pz['coordinates'];A=[a@z[row['method']] for a in X];expected=common.choose(A[0],Y[0][:,ids],wt[0],len(coords),False) if len(coords)<64 else np.arange(64);assert np.array_equal(expected,coords);A=[a[:,coords] for a in A];mx=wt[0]@A[0];my=wt[0]@Y[0][:,ids];xx=A[0]-mx;rhs=(xx*wt[0][:,None]).T@(Y[0][:,ids]-my);G=(xx*wt[0][:,None]).T@xx;betas=[la.solve(G+a*np.eye(len(coords)),rhs,assume_a='pos') for a in common.AL];loss=[np.sum(wt[1]*np.mean(((A[1]-mx)@b+my-Y[1][:,ids])**2,1)) for b in betas];b=int(np.argmin(loss));assert common.AL[b]==row['alpha'];p=((A[2]-mx)@betas[b]+my)*sd[ids]+ym[ids];err=max(err,float(np.max(np.abs(p-pz['prediction']))));assert np.array_equal(pz['truth'],m['labels'][ix[2]][:,ids]) and np.array_equal(pz['episode'],E[2]) and np.array_equal(pz['frame'],m['frame'][ix[2]]);score=float(np.sum(wt[2]*np.mean(((p-pz['truth'])/sd[ids])**2,1)));merr=max(merr,abs(score-row['normalized_mse']));count+=1
 assert err<1e-7 and merr<1e-8
 out={'status':'passed','files':count,'prediction_max_error':err,'normalized_metric_max_error':merr,'selection_alpha_source_indices':'recomputed and exact','method':'direct normal equations independently of primary SVD','code_sha256':sha(Path(__file__))};dump(O/'independent_validation.json',out);dump(O/'COMPLETE.json',out);print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--task',type=Path,required=True);p.add_argument('--old',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args();(verify if a.verify else run)(a.root,a.task,a.old,a.output)
