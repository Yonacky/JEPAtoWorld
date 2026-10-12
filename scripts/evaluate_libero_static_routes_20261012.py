"""G3 task-local static routes. Demo-equal train/tune/dev; fixed CPU protocol."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import pathlib,argparse,json,sys,time
import numpy as np
from complex_libero_pilot import sha,dump
ALPHAS=[1e-7,1e-5,.001,.01,.1,1,10,100]

def weights(ep):
 u,c=np.unique(ep,return_counts=True);lookup=dict(zip(u,c));return np.array([1/(len(u)*lookup[e]) for e in ep])
def metric(y,p,w):
 mse=float(np.sum(w*(y-p)**2));den=float(np.sum(w*(y-np.sum(w*y))**2));return {'demo_equal_mse':mse,'demo_equal_r2':1-mse/den if den>1e-14 else None}
def probe(x,y,ep,k):
 w=[weights(e) for e in ep];xm=w[0]@x[0];ym=float(w[0]@y[0]);xx=[v-xm for v in x];ys=y[0]-ym;scale=np.sqrt(w[0]@(xx[0]**2));score=abs((xx[0]/np.maximum(scale,1e-10)).T@(w[0]*ys));ids=np.argsort(score)[-k:];xx=[v[:,ids] for v in xx];g=xx[0].T@(w[0][:,None]*xx[0]);rhs=xx[0].T@(w[0]*ys);best=None
 for a in ALPHAS:
  beta=np.linalg.solve(g+a*np.eye(k),rhs);loss=float(w[1]@((xx[1]@beta+ym-y[1])**2))
  if best is None or loss<best[0]:best=(loss,a,beta)
 pred=xx[2]@best[2]+ym
 return pred,{'coordinates':ids.tolist(),'xmean':xm.tolist(),'ymean':ym,'beta':best[2].tolist(),'alpha':best[1],'tune_demo_equal_mse':best[0]}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--task',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--root',type=pathlib.Path,required=True);args=ap.parse_args();O=args.output;O.mkdir(parents=True,exist_ok=False);start=time.time()
 sys.path.insert(0,str(args.root/'core/e03'));from e03_core import optimize_joint
 import torch;torch.set_num_threads(2)
 x=np.load(args.task/'features.npz')['mean'].astype(float);data=dict(np.load(args.task/'sample_manifest.npz'));assert x.shape[1]==1280 and np.isfinite(x).all();mask=[data['split']==s for s in range(3)];X=[x[m] for m in mask];Y=[data['labels'][m].astype(float) for m in mask];ep=[data['episode'][m] for m in mask];w=weights(ep[0]);assert [len(np.unique(e)) for e in ep]==[12,4,4]
 mean=w@X[0];xc=X[0]-mean;_,s,vh=np.linalg.svd(np.sqrt(w)[:,None]*xc,full_matrices=False);assert s[63]>s[0]*1e-8;basis=vh[:64].T/s[:64];Z=[(v-mean)@basis for v in X];np.testing.assert_allclose(Z[0].T@(w[:,None]*Z[0]),np.eye(64),atol=1e-8)
 frame=data['frame'][mask[0]];pairs=np.flatnonzero((ep[0][1:]==ep[0][:-1])&(np.diff(frame)==4));assert len(pairs)>64;pw=weights(ep[0][pairs]);delta=Z[0][pairs+1]-Z[0][pairs];values,sfa=np.linalg.eigh(delta.T@(pw[:,None]*delta));lag=Z[0][pairs].T@(pw[:,None]*Z[0][pairs+1]);random=np.linalg.qr(np.random.default_rng(20261012).normal(size=(64,64)))[0];block,fit=optimize_joint(lag[None],4,[np.eye(64),sfa,random],steps=150,device='cpu')
 progress=(data['frame']/np.maximum(data['episode_length']-1,1))[:,None];representations={'white':Z,'sfa':[z@sfa for z in Z],'full_block4':[z@block for z in Z],'progress_cubic_oracle':[np.concatenate([progress[m]**j for j in [1,2,3]],axis=1) for m in mask]};params={'mean':mean,'whitening':basis,'sfa':sfa,'sfa_values':values,'block':block,'legal_train_pair_starts':pairs};np.savez_compressed(O/'transforms.npz',**params)
 predstore={'targets':Y[2],'episode':ep[2],'frame':data['frame'][mask[2]]};rows=[];errors=[]
 for method,v in representations.items():
  for k in ([4,8,64] if method!='progress_cubic_oracle' else [3]):
   for j,name in enumerate(data['target_names']):
    y=[yy[:,j] for yy in Y];pred,meta=probe(v,y,ep,k);key=f'{method}_k{k}_{name}';predstore[key]=pred;again=(v[2]-np.array(meta['xmean']))[:,meta['coordinates']]@np.array(meta['beta'])+meta['ymean'];errors.append(float(np.max(abs(again-pred))));rows.append({'method':method,'budget':k,'target':str(name),'key':key,**metric(y[2],pred,weights(ep[2])),**meta,'per_demo_mse':{str(e):float(np.mean((pred[ep[2]==e]-y[2][ep[2]==e])**2)) for e in np.unique(ep[2])}})
 np.savez_compressed(O/'predictions.npz',**predstore)
 config={'source_sha256':sha(pathlib.Path(__file__)),'inputs':{n:sha(args.task/n) for n in ['features.npz','sample_manifest.npz','data.json']},'split_demo_counts':[12,4,4],'whitening':'train demo-equal weighted PCA64, no labels','sfa':'train demo-equal legal adjacent sampled pairs actual lag4','block':'existing optimize_joint rank4 blocks,150steps, fixed identity/SFA/random initializers objective-selected train only','probes':'train demo-equal covariance top4/8/64; ridge alpha tune demo-equal; dev scoring','progress':'cubic full-episode duration privileged oracle','alphas':ALPHAS,'device':'cpu','interpretation':'each task independently fitted/evaluated; NOT zero-shot cross-task transfer; supervised coordinate diagnosis only','limitations':['Only 4 dev demos; no frame-independent confidence claim.','Raw mean1280 compressed to64; no assertion that full encoder lacks information.','No actions causality or simulator states alignment assumption.','New tasks chosen as development; not untouched confirmation.']};dump(O/'config.json',config);dump(O/'results.json',{'rows':rows,'block_fit':fit})
 saved=np.load(O/'predictions.npz');metricerr=[]
 for r in rows:
  j=list(data['target_names']).index(r['target']);m=metric(saved['targets'][:,j],saved[r['key']],weights(saved['episode']));metricerr.append(abs(m['demo_equal_mse']-r['demo_equal_mse']))
 assert max(errors+metricerr)<1e-10;dump(O/'validation.json',{'prediction_recompute_maxabs':max(errors),'saved_metric_recompute_maxabs':max(metricerr),'legal_pair_count':len(pairs),'passed':True});dump(O/'COMPLETE.json',{'seconds':time.time()-start,'status':'task-local static route comparison complete; exploratory'});print(O,flush=True)
if __name__=='__main__':main()
