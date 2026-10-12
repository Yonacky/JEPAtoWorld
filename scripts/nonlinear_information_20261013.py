"""G-6a supervised information diagnostic, fixed modest kernel ridge, CPU."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import time,json,hashlib
import numpy as np
from scipy import linalg as la
from scipy.spatial.distance import cdist,pdist
R=Path('F:/2610/CVPR27');O=R/'runs/nonlinear_information_20261013';O.mkdir(exist_ok=True);A=np.array([1e-6,1e-4,.01,.1,1,10,100]);SEED=20261013
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
cfg={'scope':'Supervised diagnostic on previously inspected data, not mechanism discovery, causality or independent confirmation','alphas':A.tolist(),'bandwidth_multipliers':[.5,1,2],'PCA':'train-only unwhitened64; all coordinates retained; no labels','linear_scaling':'one scalar sets train mean squared norm to1 for raw and PCA independently; intercept fit train','kernel':'RBF exp(-sqdist/(2*sigma^2)), sigma=train median nonzero pair distance times multiplier; kernel train centered','ridge':'K+n*alpha*I dual regression; frame-weighted fit, per-target-group equal-demo tune selection and dev scoring','shuffle':'train episode derangement by cyclic shift of seeded random episode ordering, aligned other-episode labels at normalized frame progress; preserves trajectory smoothness and may preserve progress shortcut','code_sha256':sha(Path(__file__))};dump(O/'config.json',cfg);allrows=[];checks=[];audits=[];costs=[]
inputs=[('cw',R/'runs/spatial_holdout_20261011/features',['mean','grid'],[0,1,3,7,10],{'object_xy':[0,1],'object_green':[2],'floor_red':[3],'stage_red':[4]}),('cabinet',R/'runs/libero_validation_tasks_20261013/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo',['mean'],[0,1,2,3,4],{'ee_xyz':[0,1,2],'gripper':[3,4]})]
for ds,I,views,ids,groups in inputs:
 z=np.load(I/'sample_manifest.npz');ep=z['episode'].astype(str);sp=z['split'];fr=z['frame'];leng=z['episode_length'];sets=[set(ep[sp==s]) for s in range(3)];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i));masks=[sp==s for s in range(3)];E=[ep[m] for m in masks];Yraw=[z['labels'][m][:,ids] for m in masks];ym=Yraw[0].mean(0);ys=Yraw[0].std(0);assert (ys>1e-8).all();Y=[(y-ym)/ys for y in Yraw];yshuffle=Y[0].copy();names=sorted(sets[0]);rng=np.random.default_rng(SEED);order=rng.permutation(names).tolist();mapping=dict(zip(order,order[1:]+order[:1]));progress=fr[masks[0]]/np.maximum(leng[masks[0]]-1,1)
 for dst,src in mapping.items():
  a=np.flatnonzero(E[0]==dst);b=np.flatnonzero(E[0]==src)
  for j in range(len(ids)):yshuffle[a,j]=np.interp(progress[a],progress[b],Y[0][b,j])
 np.savez_compressed(O/f'{ds}_targets.npz',truth=Yraw[2],target_mean=ym,target_std=ys,dev_episode=E[2],split=sp,episode=ep,frame=fr,train_shuffled_targets=yshuffle)
 audits.append({'dataset':ds,'source_sha256':{str(p):sha(p) for p in [I/'features.npz',I/'sample_manifest.npz']},'split_demo_counts':list(map(len,sets)),'target_names':z['target_names'][ids].tolist(),'shuffle_mapping':mapping,'label_progress_alignment':'linear interpolation within donor train demo only'})
 for view in views:
  started=time.perf_counter();raw=np.load(I/'features.npz')[view].astype(float);X=[raw[m] for m in masks];center=X[0].mean(0);X=[x-center for x in X];u,s,vh=la.svd(X[0],full_matrices=False);basis=vh[:64].T;P=[x@basis for x in X];ret=float(np.sum(s[:64]**2)/np.sum(s**2));sigma=float(np.median(pdist(P[0])));assert sigma>0;np.savez_compressed(O/f'{ds}_{view}_pca.npz',center=center,basis=basis,singular_values=s,median_distance=sigma);n=len(X[0]);fitcost=time.perf_counter()-started
  variants=[]
  for label,V in [('raw_linear',X),('pca64_linear',P)]:
   scale=np.mean(np.sum(V[0]**2,axis=1));kernels=[v@V[0].T/scale for v in V];variants.append((label,1.,kernels))
  dist=[cdist(p,P[0],'sqeuclidean') for p in P]
  for mult in [.5,1,2]:variants.append(('pca64_rbf',mult,[np.exp(-d/(2*(sigma*mult)**2)) for d in dist]))
  prepared=[]
  for method,mult,K in variants:
   km=K[0].mean(0);total=km.mean();KK=[k-k.mean(1)[:,None]-km[None,:]+total for k in K];e,v=la.eigh(KK[0]);e=np.maximum(e,0);prepared.append((method,mult,KK,e,v))
  for control,yt in [('real',Y[0]),('episode_shuffle',yshuffle)]:
   start=time.perf_counter();mean=yt.mean(0);candidates=[]
   for method,mult,K,e,v in prepared:
    rhs=v.T@(yt-mean)
    for alpha in A:
     coef=v@(rhs/(e[:,None]+n*alpha));pred=K[1]@coef+mean;loss=np.mean([np.mean((pred[E[1]==d]-Y[1][E[1]==d])**2,axis=0) for d in sorted(sets[1])],axis=0);candidates.append({'method':method,'mult':mult,'alpha':float(alpha),'coef':coef,'loss':loss,'K':K})
   for method in ['raw_linear','pca64_linear','pca64_rbf']:
    pred=np.empty_like(Y[2]);selection={};maxerr=0.;parameters={}
    for group,jj in groups.items():
     best=min([c for c in candidates if c['method']==method],key=lambda c:np.mean(c['loss'][jj]));coef=best['coef'][:,jj];pred[:,jj]=best['K'][2]@coef+mean[jj];direct=la.solve(best['K'][0]+n*best['alpha']*np.eye(n),yt[:,jj]-mean[jj],assume_a='pos');err=float(np.max(np.abs(best['K'][2]@direct+mean[jj]-pred[:,jj])));maxerr=max(maxerr,err);selection[group]={'alpha':best['alpha'],'bandwidth_multiplier':best['mult'],'tune_std_mse':float(np.mean(best['loss'][jj]))};parameters[group+'_coef']=coef
    assert maxerr<1e-6;path=O/f'{ds}_{view}_{method}_{control}.npz';np.savez_compressed(path,prediction=pred*ys+ym,mean=mean,**parameters);saved=np.load(path)['prediction'];errs=[]
    for group,jj in groups.items():
     per={d:float(np.mean((pred[E[2]==d][:,jj]-Y[2][E[2]==d][:,jj])**2)) for d in sorted(sets[2])};mse=float(np.mean(list(per.values())));rec=float(np.mean([np.mean(((saved[E[2]==d][:,jj]-Yraw[2][E[2]==d][:,jj])/ys[jj])**2) for d in sorted(sets[2])]));errs.append(abs(mse-rec));allrows.append({'dataset':ds,'view':view,'method':method,'control':control,'group':group,'dev_macro_std_mse':mse,'demo_std_mse':per,'pca64_retained_variance':ret,**selection[group]})
    checks.append({'dataset':ds,'view':view,'method':method,'control':control,'direct_prediction_max_error':maxerr,'saved_metric_max_error':max(errs)})
   costs.append({'dataset':ds,'view':view,'control':control,'pca_seconds':fitcost,'readout_selection_seconds':time.perf_counter()-start,'train_frames':n,'kernel_matrix_mib':n*n*8/2**20,'raw_width':raw.shape[1],'nonlinear_centers':n,'linear_alpha_candidates':len(A),'rbf_candidates':len(A)*3});dump(O/('selection_'+ds+'_'+view+'_'+control+'.json'),[{'method':c['method'],'mult':c['mult'],'alpha':c['alpha'],'tune_loss_per_target':c['loss'].tolist()} for c in candidates]);print(ds,view,control,'done',flush=True)
dump(O/'results.json',{'rows':allrows,'audits':audits,'costs':costs});dump(O/'validation.json',{'status':'passed','checks':checks});print('DONE',flush=True)
