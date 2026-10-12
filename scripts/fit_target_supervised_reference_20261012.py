"""N-4 matched-target supervised reference, old train/tune only; CPU float64."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse,pathlib,json,hashlib,time,platform
import numpy as np
import torch

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def projectors(q):return torch.stack([q[:,16*k:16*(k+1)]@q[:,16*k:16*(k+1)].T for k in range(3)])

def construct(x,z,U,scenes):
 records=[];cs=[];ts=[];pairlist=[]
 for i,scene in enumerate(scenes):
  donor=scenes[(i+len(scenes)//2)%len(scenes)];ids=np.where(z['scene']==scene)[0];di=np.where(z['scene']==donor)[0];X=x[ids];Y=x[di];c=z['codes'][ids];dc=z['codes'][di];assert np.array_equal(c,dc)
  assert np.array_equal(c@np.array([9,3,1]),np.arange(27))
  H=X@U;G=Y@U;offset=G[[0,26]].mean(0)-H[[0,26]].mean(0);pairlist.append({'recipient':int(scene),'donor':int(donor),'anchor_indices':[0,26]});cc=[];tt=[]
  for k in range(3):
   a,d=np.where(c[:,None,k]!=c[None,:,k]);other=[j for j in range(3) if j!=k];mask=~np.all(c[a][:,other]==c[d][:,other],axis=1);a=a[mask];d=d[mask];tc=c[a].copy();tc[:,k]=c[d,k];target=tc@np.array([9,3,1]);tfull=X[target]-X[a];den=np.sum(tfull*tfull,axis=1);assert np.all(den>1e-12)
   t=H[target]-H[a];v=G[d]-H[a]-offset;C=((v.T/den)@v-(v.T/den)@t-(t.T/den)@v)/len(v);T=(t.T/den)@t/len(v);cc.append(C);tt.append(T)
   records.append({'scene':int(scene),'donor':int(donor),'factor':k,'a':a,'d':d,'target':target,'v':v,'t':t,'den':den,'ids':ids})
  cs.append(cc);ts.append(tt)
 return records,np.mean(cs,axis=0),np.mean(ts,axis=0),pairlist

def fit(C,T,initial,penalty,steps,threshold):
 q0=np.concatenate([np.linalg.eigh(p)[1][:,-16:] for p in initial],axis=1);np.testing.assert_allclose(q0.T@q0,np.eye(48),atol=1e-8)
 q=torch.nn.Parameter(torch.tensor(q0,dtype=torch.float64));ct=torch.tensor(C);tt=torch.tensor(T);th=torch.tensor(threshold)
 hist=[];bestfeas=initial.copy();bestobj=float(np.mean(1+np.einsum('kij,kji->k',initial,C)));beststep=-1;bestraw=initial.copy();rawobj=bestobj;rawstep=-1;bestpen=initial.copy();penobj=bestobj;penstep=-1
 for step in range(steps+1):
  p=projectors(q);ret=torch.einsum('kij,kji->k',p,tt);each=1+torch.einsum('kij,kji->k',p,ct);loss=each.mean();violation=torch.relu(th-ret);cost=loss+penalty*(violation**2).mean()
  value=float(loss.detach());vv=ret.detach().numpy();pp=p.detach().numpy().copy();costval=float(cost.detach())
  if value<rawobj:rawobj=value;bestraw=pp;rawstep=step
  if costval<penobj:penobj=costval;bestpen=pp;penstep=step
  if np.all(vv>=threshold-1e-10) and value<bestobj:bestobj=value;bestfeas=pp;beststep=step
  hist.append({'step':step,'mean_train_error':value,'penalized_objective':costval,'retention':vv.tolist(),'slack':(vv-threshold).tolist(),'feasible':bool(np.all(vv>=threshold-1e-10))})
  if step==steps:break
  if q.grad is not None:q.grad.zero_()
  cost.backward()
  with torch.no_grad():
   gradient=q.grad;inner=q.T@gradient;tangent=gradient-q@((inner+inner.T)/2);gnorm=float((tangent*tangent).sum());lr=1.;accepted=False
   for backtrack in range(30):
    u,_,vh=torch.linalg.svd(q-lr*tangent,full_matrices=False);candidate=u@vh;pc=projectors(candidate);rc=torch.einsum('kij,kji->k',pc,tt);ec=(1+torch.einsum('kij,kji->k',pc,ct)).mean();ncost=ec+penalty*(torch.relu(th-rc)**2).mean()
    if float(ncost)<=costval-1e-4*lr*gnorm+1e-14:q.copy_(candidate);accepted=True;break
    lr*=.5
   hist[-1]['line_search_lr']=lr;hist[-1]['line_search_accepted']=accepted;hist[-1]['tangent_gradient_squared']=gnorm
 return {'feasible':bestfeas,'raw':bestraw,'penalized':bestpen,'last':pp},{'weight':penalty,'history':hist,'feasible_step':beststep,'raw_step':rawstep,'penalized_step':penstep,'fallback':beststep<0,'feasible_mean_train_error':bestobj}

def evaluate(x,z,U,records,models,C,T,threshold,out,domain):
 allmetrics={};audit=[];scene_cache={};order=['error','retention','projected_interference','target_nearest_correct','both_nontarget_preserved']
 for name,P in models.items():
  metrics=[]
  for r in records:
   sc=r['scene'];k=r['factor'];X=x[r['ids']];codes=z['codes'][r['ids']]
   if sc not in scene_cache:
    H=X@U;norm=np.sum(X*X,axis=1);dist=np.maximum(norm[:,None]+norm[None,:]-2*X@X.T,0);scene_cache[sc]=(H,dist)
   H,dist=scene_cache[sc];v=r['v'];t=r['t'];den=r['den'];a=r['a'];target=r['target'];d=r['d'];pred=v@P[k];ret=np.sum((t@P[k])**2,axis=1)/den;err=1+(np.sum(pred*pred,axis=1)-2*np.sum(pred*t,axis=1))/den;inter=np.sum(((v-t)@P[k])**2,axis=1)/den
   np.testing.assert_allclose(err,1-ret+inter,atol=2e-9)
   # Direct full-space audit: first 7 edits of EVERY scene/factor/model, fixed before fitting.
   direct=np.sum((X[a[:7]]+pred[:7]@U.T-X[target[:7]])**2,axis=1)/den[:7];audit.append(float(np.max(abs(direct-err[:7]))))
   distances=dist[a]+2*np.einsum('ni,nji->nj',pred,H[a,None]-H[None])+np.sum(pred*pred,axis=1)[:,None];nearest=distances.argmin(1);other=[j for j in range(3) if j!=k];correct=codes[nearest,k]==codes[d,k];keep=np.all(codes[nearest][:,other]==codes[a][:,other],axis=1)
   metrics.append([err.mean(),ret.mean(),inter.mean(),correct.mean(),keep.mean()])
  m=np.array(metrics).reshape(-1,3,5);allmetrics[name]=m
  if name!='no_edit':
   np.testing.assert_allclose(m.mean(0)[:,0],1+np.einsum('kij,kji->k',P,C),atol=2e-9);np.testing.assert_allclose(m.mean(0)[:,1],np.einsum('kij,kji->k',P,T),atol=2e-9)
 np.savez_compressed(out/f'{domain}_scene_metrics.npz',**allmetrics)
 return {'metric_order':order,'models':{n:{'factor_means':v.mean(0).tolist(),'mean_error':float(v[:,:,0].mean()),'retention_slack_against_train_threshold':(v[:,:,1].mean(0)-threshold).tolist()} for n,v in allmetrics.items()},'direct_full_space_maxabs':max(audit),'direct_full_space_edits_checked':len(audit)*7}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--steps',type=int,default=300);args=ap.parse_args();R=args.study;O=args.output;O.mkdir(parents=True,exist_ok=True);assert not (O/'COMPLETE.json').exists(),'preserve existing complete run'
 torch.set_num_threads(2);start=time.time();paths=[R/'pilot/dataset.npz',R/'features/features.npz',R/'step2b/fitted.npz'];missing=[str(p) for p in paths if not p.exists()];assert not missing,missing
 z=dict(np.load(paths[0]));x=np.load(paths[1])['grid'].astype(float);b=dict(np.load(paths[2]));U=b['U'].astype(float);initial=b['joint_0'].astype(float)
 np.testing.assert_allclose(U.T@U,np.eye(64),atol=1e-8)
 for k,p in enumerate(initial):
  np.testing.assert_allclose(p@p,p,atol=1e-8);assert np.linalg.matrix_rank(p,tol=1e-7)==16
  for j in range(k):np.testing.assert_allclose(p@initial[j],0,atol=1e-8)
 config={'steps':args.steps,'dtype':'float64','device':'cpu','optimizer':'Riemannian projected gradient; polar retraction; Armijo c=1e-4, initial step1, max30 halvings','penalty_weights':[0,1,10,100],'penalty':'weight * mean(relu(threshold-retention)^2)','threshold_fraction':.95,'factors':['position','color','background'],'selection':'train feasible checkpoint minimum unpenalized mean error; all weights reported; tune never selects','scope':'matched-target supervised reference, not unsupervised discovery or strict upper bound; reused old train32/tune8','source_sha256':sha(pathlib.Path(__file__)),'inputs':{str(p.relative_to(R)):sha(p) for p in paths},'python':platform.python_version(),'torch':torch.__version__}
 dump(O/'config.json',config)
 train,C,T,pairs=construct(x,z,U,list(range(32)));threshold=.95*np.einsum('kij,kji->k',initial,T);np.savez_compressed(O/'training_statistics.npz',C=C,T=T,threshold=threshold);dump(O/'train_pairs.json',pairs)
 models={'joint0':initial,'no_edit':np.zeros_like(initial)};fithistory={}
 for w in [0,1,10,100]:
  mm,meta=fit(C,T,initial,w,args.steps,threshold);fithistory[str(w)]=meta
  for name,p in mm.items():models[f'penalty{w}_{name}']=p
  print('weight',w,'feasible_step',meta['feasible_step'],'error',meta['feasible_mean_train_error'],flush=True)
 dump(O/'fit.json',fithistory);np.savez_compressed(O/'models.npz',U=U,**models)
 for name,pp in models.items():
  if name=='no_edit':continue
  for k,p in enumerate(pp):
   np.testing.assert_allclose(p@p,p,atol=1e-9);np.testing.assert_allclose(p,p.T,atol=1e-9);assert np.linalg.matrix_rank(p,tol=1e-7)==16
   for j in range(k):np.testing.assert_allclose(p@pp[j],0,atol=1e-9)
  if name.endswith('_feasible'):assert np.min(np.einsum('kij,kji->k',pp,T)-threshold)>-1e-10
 result={'train':evaluate(x,z,U,train,models,C,T,threshold,O,'train')}
 tune,ct,tt,pairs=construct(x,z,U,list(range(32,40)));dump(O/'tune_pairs.json',pairs);result['tune']=evaluate(x,z,U,tune,models,ct,tt,threshold,O,'tune');dump(O/'results.json',result)
 dump(O/'COMPLETE.json',{'status':'fit and numerical audit completed; exploratory supervised reference only','seconds':time.time()-start,'source_sha256':sha(pathlib.Path(__file__))});print('COMPLETE',time.time()-start,flush=True)
if __name__=='__main__':main()
