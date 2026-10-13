"""G7g bounded similarity-block edit fitting, supervised matched targets."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import json,hashlib,argparse
from pathlib import Path
import numpy as np
from scipy import linalg as la
import torch
import fit_target_supervised_reference_20261012 as old

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def project(W):
 inv=torch.linalg.inv(W);return torch.stack([W[:,k*16:(k+1)*16]@inv[k*16:(k+1)*16] for k in range(3)])
def matrices(W):return np.array([W[:,k*16:(k+1)*16]@la.inv(W)[k*16:(k+1)*16] for k in range(4)])
def statistics(x,z,U,scenes):
 rec,_,_,pairs=old.construct(x,z,U,scenes);S=[];R=[];L=[]
 for k in range(3):
  rr=[r for r in rec if r['factor']==k];S.append(np.mean([(r['v'].T/r['den'])@r['v']/len(r['v']) for r in rr],0));R.append(np.mean([(r['v'].T/r['den'])@r['t']/len(r['v']) for r in rr],0));lc=[]
  for sc in scenes:
   ids=np.flatnonzero(z['scene']==sc);c=z['codes'][ids];X=x[ids];H=X@U;jc=[]
   for j in range(3):
    if j==k:continue
    other=[h for h in range(3) if h!=j];a,b=np.where((c[:,None,j]!=c[None,:,j])&np.all(c[:,None,other]==c[None,:,other],axis=2));v=H[b]-H[a];den=np.sum((X[b]-X[a])**2,1);assert np.all(den>1e-12);jc.append((v.T/den)@v/len(v))
   lc.append(np.mean(jc,0))
  L.append(np.mean(lc,0))
 return rec,np.array(S),np.array(R),np.array(L),pairs

def fit(W0,S,R,L,family,weight):
 W=torch.tensor(W0,dtype=torch.float64,requires_grad=True);S,R,L=map(torch.tensor,[S,R,L]);hist=[]
 def cost(V):
  P=project(V);error=1+sum(torch.trace(P[k].T@S[k]@P[k])-2*torch.sum(P[k]*R[k]) for k in range(3))/3;leak=sum(torch.trace(P[k].T@L[k]@P[k]) for k in range(3))/3;return error+weight*leak,error,leak
 for step in range(201):
  value,err,leak=cost(W);hist.append({'step':step,'cost':float(value.detach()),'error':float(err.detach()),'leak':float(leak.detach()),'condition':float(torch.linalg.cond(W).detach())})
  if step==200:break
  if W.grad is not None:W.grad.zero_()
  value.backward();g=W.grad.detach();lr=1.;accepted=False
  with torch.no_grad():
   for bt in range(30):
    u,s,vh=torch.linalg.svd(W-lr*g);s=torch.ones_like(s) if family=='orth' else torch.clamp(s,2**-.5,2**.5);candidate=(u*s)@vh;delta=candidate-W;val=cost(candidate)[0]
    if float(val)<=float(value)-1e-4*float((delta*delta).sum())/lr+1e-14:W.copy_(candidate);accepted=True;break
    lr*=.5
  hist[-1].update(accepted=accepted,lr=lr)
  if not accepted:break
 return W.detach().numpy(),hist

def evaluate(x,z,U,rec,L,models,O,domain):
 metrics={};predsaved={};maxerr=0.;cache={}
 for name,(Ps,anchors) in models.items():
  rows=[]
  for ri,r in enumerate(rec):
   sc,k=r['scene'],r['factor'];X=x[r['ids']];c=z['codes'][r['ids']];H=X@U
   if sc not in cache:
    n=np.sum(X*X,1);cache[sc]=np.maximum(n[:,None]+n[None,:]-2*X@X.T,0)
   v=r['v'].copy()
   if not anchors:
    di=np.flatnonzero(z['scene']==r['donor']);G=x[di]@U;v=G[r['d']]-H[r['a']]
   pred=v@Ps[k];t=r['t'];den=r['den'];error=1+(np.sum(pred**2,1)-2*np.sum(pred*t,1))/den;ret=np.sum((t@Ps[k])**2,1)/den
   direct=np.sum((X[r['a'][:7]]+pred[:7]@U.T-X[r['target'][:7]])**2,1)/den[:7];maxerr=max(maxerr,float(np.max(abs(direct-error[:7]))));predsaved[name+f'_s{sc}_f{k}']=pred[:7]
   distances=cache[sc][r['a']]+2*np.einsum('ni,nji->nj',pred,H[r['a'],None]-H[None])+np.sum(pred**2,1)[:,None];nearest=distances.argmin(1);other=[j for j in range(3) if j!=k];rows.append([error.mean(),ret.mean(),np.mean(c[nearest,k]==c[r['d'],k]),np.mean(np.all(c[nearest][:,other]==c[r['a']][:,other],1))])
  rows=np.array(rows).reshape(-1,3,4);leak=[float(np.trace(Ps[k].T@L[k]@Ps[k])) for k in range(3)];metrics[name]={'factor_means':rows.mean(0).tolist(),'mean_error':float(rows[:,:,0].mean()),'cross_factor_leakage':leak,'selection_score':float(rows[:,:,0].mean()+np.mean(leak))};np.savez_compressed(O/(domain+'_'+name+'_scene.npz'),metrics=rows)
 np.savez_compressed(O/(domain+'_prediction_samples.npz'),**predsaved);assert maxerr<1e-8;return {'metric_order':['full_space_relative_error','target_projected_energy_ratio_not_recovery','target_nearest_correct','both_nontarget_preserved'],'models':metrics,'direct_full_space_error_max':maxerr}

def main(a):
 torch.set_num_threads(2);O=a.output;O.mkdir(parents=True,exist_ok=False);R=a.study;z=dict(np.load(R/'pilot/dataset.npz'));x=np.load(R/'features/features.npz')['grid'].astype(float);f=np.load(R/'step2b/fitted.npz');U=f['U'].astype(float);initial=f['joint_0'].astype(float);Q=np.concatenate([la.eigh(p)[1][:,-16:] for p in initial],1);W0=np.concatenate([Q,la.null_space(Q.T)],1);np.testing.assert_allclose(W0.T@W0,np.eye(64),atol=1e-8)
 train,S,T,L,pairs=statistics(x,z,U,list(range(32)));dump(O/'train_pairs.json',pairs);np.savez_compressed(O/'train_statistics.npz',S=S,R=T,L=L);Ws={};models={'no_edit':(np.zeros_like(initial),True),'joint0_uncalibrated':(initial,False),'joint0_twoanchors':(initial,True)};history={};invariants={}
 for family in ['orth','oblique']:
  for weight in [0,1]:
   name=f'{family}_lambda{weight}';W,hist=fit(W0,S,T,L,family,weight);Ws[name]=W;history[name]=hist;pp=matrices(W);models[name]=(pp[:3],True);sv=la.svdvals(W);idem=max(float(np.max(abs(p@p-p))) for p in pp);mut=max(float(np.max(abs(pp[i]@pp[j]))) for i in range(4) for j in range(4) if i!=j);summ=float(np.max(abs(pp.sum(0)-np.eye(64))));assert max(idem,mut,summ)<1e-9 and sv.max()<=2**.5+1e-10 and sv.min()>=2**-.5-1e-10;invariants[name]={'condition':float(sv[0]/sv[-1]),'singular_min':float(sv[-1]),'singular_max':float(sv[0]),'idempotence':idem,'annihilation':mut,'sum4_identity':summ,'projector_spectral_norm':[float(la.norm(p,2)) for p in pp],'isotropic_noise_mean_squared_gain':[float(np.sum(p*p)/64) for p in pp],'symmetry_error':[float(np.max(abs(p-p.T))) for p in pp]};print(name,hist[-1],flush=True)
 np.savez_compressed(O/'models.npz',U=U,W0=W0,**Ws);dump(O/'history.json',history);dump(O/'invariants.json',invariants);result={'train':evaluate(x,z,U,train,L,models,O,'train')};tune,_,_,Lt,pp=statistics(x,z,U,list(range(32,40)));dump(O/'tune_pairs.json',pp);result['tune']=evaluate(x,z,U,tune,Lt,models,O,'tune');selection={family:min([f'{family}_lambda{w}' for w in [0,1]],key=lambda n:result['tune']['models'][n]['selection_score']) for family in ['orth','oblique']};dump(O/'selection.json',selection)
 for domain in ['id','ood']:
  zz=dict(np.load(R/f'tworef_{domain}_data/dataset.npz'));xx=np.load(R/f'tworef_{domain}_features/features.npz')['grid'].astype(float);scenes=sorted(np.unique(zz['scene']).tolist());rr,_,_,ll,pp=statistics(xx,zz,U,scenes);dump(O/(domain+'_pairs.json'),pp);result[domain]=evaluate(xx,zz,U,rr,ll,models,O,domain);print(domain,'done',flush=True)
 dump(O/'results.json',result);dump(O/'config.json',{'scope':'Supervised matched edit optimization, reused ID/OOD exploration; not ICA or unsupervised','steps':200,'weights':[0,1],'condition_cap':2,'selection':'tune edit error+cross factor leakage; all candidates reported','input_hashes':{str(p):sha(p) for p in [R/'pilot/dataset.npz',R/'features/features.npz',R/'step2b/fitted.npz',Path(__file__),Path(old.__file__)]}})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
