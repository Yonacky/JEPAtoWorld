"""S-2 fixed-target demo learning curve; CPU, no test selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy import linalg as la
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--original',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();I=args.input;OLD=args.original;O=args.output;O.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def ordered(a):return list(dict.fromkeys(a.tolist()))
A=np.array([1e-6,1e-4,.01,.1,1,10,100]);SEEDS=[20261012,20261013,20261014];D=64
z=np.load(I/'sample_manifest.npz');old=np.load(OLD/'sample_manifest.npz');x=np.load(I/'features.npz')['mean'].astype(float);y=z['labels'];ep=z['episode'];sp=z['split'];fr=z['frame'];le=z['episode_length'];orig=ordered(old['episode'][old['split']==0]);assert len(orig)==12
meta=json.loads((I/'data.json').read_text(encoding='utf8'));alltrain=meta.get('training_order',meta.get('train_order'));assert alltrain is not None,'Missing explicit training_order';assert set(alltrain)==set(ep[sp==0]) and alltrain[:12]==orig;new=alltrain[12:];assert len(new)==30 and len(alltrain)==42
for split in [1,2]:assert set(ep[sp==split])==set(old['episode'][old['split']==split])
# Enforce unchanged old rows/labels; extraction numerical consistency is recorded separately.
oldx=np.load(OLD/'features.npz')['mean'];lookup={(str(e),int(f)):i for i,(e,f) in enumerate(zip(ep,fr))};mapold=np.array([lookup[(str(e),int(f))] for e,f in zip(old['episode'],old['frame'])]);assert np.array_equal(y[mapold],old['labels']);assert np.array_equal(sp[mapold],old['split']);feature_difference=float(np.max(np.abs(x[mapold]-oldx)))
sets=[set(ep[sp==s]) for s in range(3)];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i));windows={}
for demo in ordered(ep):
 q=np.flatnonzero(ep==demo);assert np.all(np.diff(fr[q])==4) and len(np.unique(sp[q]))==1;windows[demo]=np.array([q[j-3:j+3] for j in range(3,len(q)-2)])
def rows(demos):return np.concatenate([windows[d] for d in demos])
def pca64(data):
 m=data.mean(0);_,s,v=la.svd(data-m,full_matrices=False);assert s[63]>1e-8;return m,v[:64].T/(s[:64]/np.sqrt(len(data)))
# Frozen target uses original cached features, not newly extracted rounding differences.
fm,fb=pca64(oldx[old['split']==0].astype(float));target=(x-fm)@fb
q0=rows(orig);tune=rows(ordered(ep[sp==1]));dev=rows(ordered(ep[sp==2]));devdemo=ep[dev[:,3]];tunedemo=ep[tune[:,3]]
def raw(q):return np.column_stack([y[q[:,3]],y[q[:,3]]-y[q[:,2]],target[q[:,4]],target[q[:,5]]])
fixed=raw(q0);ym=fixed.mean(0);ys=fixed.std(0);assert np.all(ys>1e-8)
groups={'position':slice(0,3),'gripper':slice(3,5),'backward_displacement':slice(5,8),'gripper_change':slice(8,10),'future4frames':slice(10,74),'future8frames':slice(74,138)}
# Exactly source order for original12/new30, shuffled only at size24; canonical size42.
subsets=[('n12',orig),('n42',sorted(alltrain,key=lambda s:int(s.rsplit('_',1)[1])))]
for seed in SEEDS:subsets.append((f'n24_seed{seed}',orig+np.random.default_rng(seed).permutation(new)[:12].tolist()))
config={'scope':'S-2 repeated fixed dev learning curve, not independent confirmation','source_sha256':{str(p):sha(p) for p in [I/'features.npz',I/'sample_manifest.npz',OLD/'features.npz',OLD/'sample_manifest.npz']},'code_sha256':sha(Path(__file__)),'original12':orig,'additional_source_order':new,'subsets':dict(subsets),'seeds':SEEDS,'original_old_feature_max_error':feature_difference,'target':'Original12 cached train-frame whitenedPCA64; frozen for all train sizes, used also as RRR target for comparability','state_scale':'Original12 common-window target std frozen; physical-unit MSE also reported','rrr_lambda':.01,'alphas':A.tolist(),'method_widths':[64,256,64,64],'tune_metric':'equal-demo mean standardized MSE per group','fit':'frame-weighted train ridge/PCA, as G1; selection/report demo equal weight','labels':'Only supervised readout; no state labels in input transforms','counts':{'tune':len(tune),'dev':len(dev)},'future_steps_original_frames':[4,8]};dump(O/'config.json',config)
np.savez_compressed(O/'fixed_target_truth.npz',pca_center=fm,pca_basis=fb,target_mean=ym,target_std=ys,truth=raw(dev),dev_demo=devdemo,dev_rows=dev,tune_rows=tune)
results=[];checks=[];predictions={}
for label,demos in subsets:
 train=rows(demos);ix=[train,tune,dev];bm,bb=pca64(x[np.isin(ep,demos)]);f=(x-bm)@bb;H=[np.concatenate([f[q[:,3-j]] for j in range(4)],axis=1)/2 for q in ix];hm=H[0].mean(0);HH=[h-hm for h in H];Yraw=[raw(q) for q in ix];Y=[(a-ym)/ys for a in Yraw];F=[target[q[:,4:6]].reshape(len(q),128) for q in ix];G=HH[0].T@HH[0]/len(train);C=HH[0].T@(F[0]-F[0].mean(0))/len(train);e,pb=la.eigh(G);B=la.solve(G+.01*np.eye(256),C,assume_a='pos');e,v=la.eigh(C.T@B);rb,_=la.qr(B@v[:,-64:],mode='economic');methods={'current64':[f[q[:,3]] for q in ix],'history256':H,'history_pca64':[h@pb[:,-64:] for h in HH],'history_rrr64':[h@rb for h in HH]}
 np.savez_compressed(O/f'{label}_transforms.npz',base_center=bm,base=bb,history_center=hm,pca=pb[:,-64:],rrr=rb,train_rows=train,train_demos=demos)
 for name,V in methods.items():
  vm=V[0].mean(0);my=Y[0].mean(0);VV=[v-vm for v in V];g=VV[0].T@VV[0]/len(train);c=VV[0].T@(Y[0]-my)/len(train);e,v=la.eigh(g);rhs=v.T@c;e=np.maximum(e,0);ws=[];loss=[]
  for a in A:
   w=v@(rhs/(e[:,None]+a));ws.append(w);p=VV[1]@w+my;loss.append(np.mean([np.mean((p[tunedemo==d]-Y[1][tunedemo==d])**2,axis=0) for d in np.unique(tunedemo)],axis=0))
  loss=np.array(loss);choice=np.zeros(138,dtype=int)
  for sl in groups.values():choice[sl]=np.argmin(loss[:,sl].mean(1))
  w=np.column_stack([ws[choice[j]][:,j] for j in range(138)]);pred=VV[2]@w+my;err=0.
  for ai in np.unique(choice):
   cols=np.flatnonzero(choice==ai);ww=la.solve(g+A[ai]*np.eye(len(g)),c[:,cols],assume_a='pos');err=max(err,float(np.max(np.abs(VV[2]@ww+my[cols]-pred[:,cols]))))
  assert err<1e-6;np.savez_compressed(O/f'{label}_{name}.npz',prediction_physical=pred*ys+ym,weights=w,feature_center=vm,train_target_center=my,alpha=A[choice],tune_mse=loss)
  predictions[(label,name)]=pred;checks.append({'subset':label,'method':name,'direct_error':err,'rank':int(np.sum(e>max(e.max()*1e-9,1e-12)))})
  for group,sl in groups.items():
   phys=pred[:,sl]*ys[sl]+ym[sl];per={d:float(np.mean((pred[devdemo==d,sl]-Y[2][devdemo==d,sl])**2)) for d in np.unique(devdemo)};pu={d:float(np.mean((phys[devdemo==d]-Yraw[2][devdemo==d,sl])**2)) for d in np.unique(devdemo)};results.append({'subset':label,'ntrain':len(demos),'method':name,'target':group,'macro_std_mse':float(np.mean(list(per.values()))),'macro_physical_mse':float(np.mean(list(pu.values()))),'demo_std_mse':per,'demo_physical_mse':pu,'alpha':float(A[choice[sl.start]]),'train_windows':len(train)})
  print(label,name,'done',flush=True)
# Independent disk-only metric recomputation.
errs=[]
for row in results:
 pp=np.load(O/f"{row['subset']}_{row['method']}.npz")['prediction_physical'];sl=groups[row['target']];tt=raw(dev);mm=np.mean([np.mean(((pp[devdemo==d,sl]-tt[devdemo==d,sl])/ys[sl])**2) for d in np.unique(devdemo)]);errs.append(abs(mm-row['macro_std_mse']))
assert max(errs)<1e-10
comparisons=[];rng=np.random.default_rng(20261012);nd=len(np.unique(devdemo));boots=rng.integers(nd,size=(1000,nd))
for method in methods:
 for group in groups:
  a=next(r for r in results if r['subset']=='n12' and r['method']==method and r['target']==group);b=next(r for r in results if r['subset']=='n42' and r['method']==method and r['target']==group);delta=np.array(list(a['demo_std_mse'].values()))-np.array(list(b['demo_std_mse'].values()));comparisons.append({'method':method,'target':group,'n12_minus_n42_positive_improves':float(delta.mean()),'paired_demo_ci95':np.percentile(delta[boots].mean(1),[2.5,97.5]).tolist()})
dump(O/'results.json',{'rows':results,'n12_vs_n42':comparisons});dump(O/'validation.json',{'status':'passed','checks':checks,'saved_metric_max_error':max(errs),'unchanged_demo_split':True,'fixed_target':True});print('DONE',flush=True)
