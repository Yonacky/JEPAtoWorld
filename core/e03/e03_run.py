import os
os.environ['OPENBLAS_NUM_THREADS']='6';os.environ['OMP_NUM_THREADS']='6'
import pathlib,json,time,hashlib
import numpy as np
from scipy import linalg as la
import torch
import torch.nn.functional as F
from e03_core import *
from device_utils import activate_device

R=pathlib.Path(__file__).parent;cfg=json.loads((R/'protocol.json').read_text());selftest=json.loads((R/'selftest.json').read_text())
assert selftest['all_passed'] and selftest['core_sha256']==hashlib.sha256((R/'e03_core.py').read_bytes()).hexdigest()
torch.set_num_threads(6);device=activate_device();start=time.time();rows=[];probes=[];structures=[];checks=[]
for n in ['cache','transforms','models','probes']:
 # models is the pre-existing read-only checkpoint symlink; prediction fits get a separate folder.
 if n!='models':(R/n).mkdir(exist_ok=True)
(R/'predictor_fits').mkdir(exist_ok=True)
def atomic(p,obj):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
codehash={p.name:sha(p) for p in R.glob('*.py')};atomic(R/'run_source_snapshot.json',{'code':codehash,'protocol_sha256':sha(R/'protocol.json'),'start_unix':start,'device':str(device)})
for ds in cfg['datasets']:
 data=np.load(R/'prepared'/f'{ds}.npz');allY,groups=encode_targets(ds,data['latents'],data['keys'].tolist());atomic(R/f'{ds}_factor_schema.json',groups)
 for enc in cfg['encoders']:
  if enc=='pixels':
   im=torch.from_numpy(data['images']).permute(0,3,1,2).float()/255;allX=F.interpolate(im,size=(16,16),mode='bilinear',align_corners=False,antialias=True).flatten(1).numpy();del im
  else:
   fp=R/'features'/f'{ds}_{enc}.npz';fm=json.loads(fp.with_suffix('.json').read_text());assert fm['cache_sha256']==sha(fp);allX=np.load(fp)['mean']
  assert np.isfinite(allX).all() and len(allX)==3*10497
  for rep in range(3):
   tag=f'{ds}_{enc}_r{rep}';ix=[np.arange(*cfg[k])+rep*10497 for k in ['train','tune','development_holdout']];source=[data['source_indices'][i] for i in ix]
   assert source[0][-1]<source[1][0]<source[1][-1]<source[2][0]
   for idx in source:
    for h in cfg['forecast_horizons']:pair_indices(idx,h)
   center,W,ev=pca_whiten(allX[ix[0]],cfg['dim']);X=[(allX[i].astype(np.float64)-center)@W for i in ix];Y=[allY[i] for i in ix];n=cfg['dim'];eye=np.eye(n)
   np.savez_compressed(R/'cache'/f'{tag}.npz',train=X[0],tune=X[1],dev=X[2],target_train=Y[0],target_tune=Y[1],target_dev=Y[2],source_train=source[0],source_tune=source[1],source_dev=source[2],center=center,pca=W,eigenvalues=ev)
   C=lag_cov(X[0],cfg['lags']);Cs=(C+C.transpose(0,2,1))/2;delta=np.diff(X[0],axis=0);_,sfa=la.eigh(delta.T@delta/len(delta));rnd=np.random.default_rng(0);randomQ=la.qr(rnd.normal(size=(n,n)))[0]
   xp=X[0][rnd.permutation(len(X[0]))];d=np.diff(xp,axis=0);_,shuffle=la.eigh(d.T@d/len(d));A,ax,ay=ridge_fit(X[0][:-1],X[0][1:],1e-4);qs,ts,natural=schur_basis(A)
   U,sv,Vh=la.svd(C[0],full_matrices=False);initials=[eye,sfa,qs]
   methods=[('white',eye,eye,{}),('random_Q',randomQ,randomQ,{}),('sfa',sfa,sfa,{}),('sfa_time_shuffle',shuffle,shuffle,{})]
   for name,operators,b in [('sym_block4',Cs,4),('full_block2',C,2),('full_block4',C,4),('full_block8',C,8)]:
    q,info=optimize_joint(operators,b,initials,steps=250,device=device);methods.append((name,q,q,info))
   methods += [('real_schur',qs,qs,{'natural_group_sizes':[len(g) for g in natural],'reconstruction_error':float(np.max(np.abs(A-qs@ts@qs.T))),'off_natural_block_energy':float(np.sum((ts*mask_for(natural,n))**2)/np.sum(ts*ts))}),('lag_svd_dual',U,Vh.T,{'singular_values':sv.tolist(),'scope':'past and future use different orthogonal bases'})]
   dense_mse={};dense_probe=None
   for name,ql,qr,info in methods:
    assert np.max(np.abs(ql.T@ql-eye))<1e-8 and np.max(np.abs(qr.T@qr-eye))<1e-8
    np.savez_compressed(R/'transforms'/f'{tag}_{name}.npz',past_basis=ql,future_basis=qr,training_lag_cov=C,one_step_operator=A)
    sx=[a@ql for a in X];sy=[a@qr for a in X];devC=lag_cov(X[2],cfg['lags']);energy={}
    for b in [1,2,4,8]:
     mask=mask_for(groups_fixed(n,b),n);energy[f'full_dev_offblock_b{b}']=float(np.sum(((ql.T@devC@qr)*mask)**2)/np.sum(devC**2))
    structures.append(dict(dataset=ds,encoder=enc,replica=rep,method=name,antisymmetric_train_ratio=float(np.sum(((C-C.transpose(0,2,1))/2)**2)/np.sum(C*C)),**energy,**info))
    for budget in [1,4,8,'dense']:
     p,selected=probe(sx,Y,groups,budget);path=R/'probes'/f'{tag}_{name}_k{budget}.npz';np.savez_compressed(path,prediction=p,source_indices=source[2])
     probes.append(dict(dataset=ds,encoder=enc,replica=rep,method=name,budget=budget,metrics=factor_metrics(Y[2],p,groups),selected=selected,predictions=str(path.relative_to(R))))
     if budget=='dense':
      if dense_probe is None:dense_probe=p
      err=float(np.max(np.abs(p-dense_probe)));assert err<1e-6;checks.append({'kind':'dense_probe_invariance','tag':tag,'method':name,'maxabs':err})
    for h in cfg['forecast_horizons']:
     xx=[a[:-h] for a in sx];yy=[a[h:] for a in sy];truth=X[2][h:];persist=float(np.mean((X[2][:-h]-truth)**2));meanbase=float(np.mean((X[0][h:].mean(0)-truth)**2))
     groupings=[(str(b),groups_fixed(n,b)) for b in [1,2,4,8,128]]
     if name=='real_schur':groupings.append(('natural_schur',natural))
     for grouping,gs in groupings:
      p,fit=forecast_fit(xx,yy,gs);canonical=p@qr.T;perframe=np.mean((canonical-truth)**2,axis=1);mse=float(perframe.mean());path=R/'predictor_fits'/f'{tag}_{name}_h{h}_b{grouping}.npz'
      np.savez_compressed(path,weight=fit['weight'],input_mean=fit['input_mean'],output_mean=fit['output_mean'],per_frame_mse=perframe,source_indices=source[2][h:])
      rows.append(dict(dataset=ds,encoder=enc,replica=rep,method=name,horizon=h,grouping=grouping,alpha=fit['alpha'],parameters=fit['parameters'],tune_mse=fit['tune_mse'],dev_mse=mse,persistence_mse=persist,mean_baseline_mse=meanbase,skill_vs_persistence=1-mse/persist,skill_vs_mean=1-mse/meanbase,model_file=str(path.relative_to(R))))
      if grouping=='128':
       if h not in dense_mse:dense_mse[h]=mse
       err=abs(mse-dense_mse[h]);assert err<1e-8;checks.append({'kind':'dense_forecast_invariance','tag':tag,'method':name,'horizon':h,'maxabs':err})
    atomic(R/'probes.json',probes);atomic(R/'forecasts.json',rows);atomic(R/'structures.json',structures);atomic(R/'invariance_checks.json',checks)
    print('DONE',tag,name,'forecast_rows',len(rows),flush=True)
   atomic(R/'progress.json',{'completed':tag,'seconds':time.time()-start})
assert len(rows)==24*(10*15+3) and len(probes)==24*10*4
assert codehash=={p.name:sha(p) for p in R.glob('*.py')},'code changed during run'
atomic(R/'COMPLETE.json',{'forecast_rows':len(rows),'probe_rows':len(probes),'transform_fits':len(structures),'seconds':time.time()-start,'phase':cfg['phase'],'official_test_access':False});print('COMPLETE',len(rows),len(probes),flush=True)
