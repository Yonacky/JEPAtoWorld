"""G5b supervised conditional physical readout; not unique parameter identification."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json,hashlib,argparse
import numpy as np
from scipy import linalg as la
ALPHAS=[1e-5,.001,.1,1.,10.]
TIMES=[0,1,5,10,20,30,60,90,120]
VERSIONS=['t0','t30','t120','concat_0_30_120','mean9']
FACTORS=['mass','friction','force']
TIE=1e-10

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def predict_class(scores):return np.argmax(scores>=scores.max(1,keepdims=True)-TIE,axis=1)
def metrics(y,p,scene):
 conf=np.zeros((3,3),dtype=int);np.add.at(conf,(y,p),1);bacc=float(np.mean(np.diag(conf)/conf.sum(1)));per=[{'scene':int(s),'accuracy':float(np.mean(y[scene==s]==p[scene==s]))} for s in np.unique(scene)];return {'balanced_accuracy':bacc,'confusion':conf.tolist(),'per_scene':per}
def load(data,features):
 validation=json.loads((data.parent/'validation.json').read_text());assert validation['status']=='PASS' and validation['dataset_sha256']==sha(data), 'Production validation/hash failed'
 z=np.load(data);m={k:z[k] for k in ['split','scene','factor','level','episode','frame','actual_states','state_names','labels']};f=np.load(features);fm=np.load(features.parent/'sample_manifest.npz');
 for key in ['split','episode','frame','labels']:assert np.array_equal(fm[key],m[key]), ('Feature/source row alignment',key)
 arrays={k:f[k].astype(float) for k in ['mean','grid']};assert arrays['mean'].shape==(2520,1280) and arrays['grid'].shape==(2520,5120);assert list(m['state_names'])==['x','y','z','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz'];st=m['actual_states'];assert st.shape[0]==2520 and st.shape[-1]==13 and st.shape[1]==1;arrays['state_posvel']=st[:,0,[0,1,2,7,8,9]].astype(float);assert all(np.isfinite(v).all() for v in arrays.values());assert set(m['scene'])==set(range(40));assert [len(np.unique(m['scene'][m['split']==i])) for i in range(3)]==[24,8,8]
 lookup={};t0=[]
 for scene in range(40):
  ix=np.flatnonzero(m['scene']==scene);assert len(ix)==63 and len(np.unique(m['split'][ix]))==1 and int(m['split'][ix[0]])==(0 if scene<24 else 1 if scene<32 else 2);branches=np.unique(m['episode'][ix]);assert len(branches)==7
  for ep in branches:
   ids=ix[m['episode'][ix]==ep];ids=ids[np.argsort(m['frame'][ids])];assert np.array_equal(m['frame'][ids],TIMES);factor=str(m['factor'][ids[0]]);level=int(m['level'][ids[0]]);assert np.all(m['factor'][ids]==factor) and np.all(m['level'][ids]==level);key=(scene,factor,level);assert key not in lookup;lookup[key]=ids
  zero=ix[m['frame'][ix]==0];assert len(zero)==7
  for source,A in arrays.items():
   diff=float(np.max(np.abs(A[zero]-A[zero[0]])));scale=max(1.,float(np.max(np.abs(A[zero]))));t0.append({'scene':scene,'source':source,'max_abs_difference':diff,'scale':scale,'relative_max_difference':diff/scale,'exact':bool(np.array_equal(A[zero],np.repeat(A[zero[:1]],7,axis=0)))})
   assert diff<=2e-6*scale, ('t0 feature mismatch',scene,source,diff)
  assert np.array_equal(z['images'][zero],np.repeat(z['images'][zero[:1]],7,axis=0))
 return m,arrays,lookup,t0

def build(m,A,lookup,factor,version):
 rows=[];labels=[];scenes=[];splits=[];sources=[]
 for s in range(40):
  for level in range(3):
   ix=lookup[s,'baseline',1] if level==1 else lookup[s,factor,level];values=A[ix]
   inp=values[TIMES.index(int(version[1:]))] if version.startswith('t') else values[[0,5,8]].ravel() if version=='concat_0_30_120' else values.mean(0)
   rows.append(inp);labels.append(level);scenes.append(s);splits.append(int(m['split'][ix[0]]));sources.append(ix)
 return np.array(rows),np.array(labels),np.array(scenes),np.array(splits),np.array(sources)

def run(data,features,O):
 O.mkdir(parents=True,exist_ok=True);m,arrays,lookup,t0=load(data,features);cfg={'data':str(data),'features':str(features),'input_hashes':{str(data):sha(data),str(features):sha(features),str(features.parent/'sample_manifest.npz'):sha(features.parent/'sample_manifest.npz')},'code_sha256':sha(Path(__file__)),'factors':FACTORS,'versions':VERSIONS,'sources':list(arrays),'pca':'Train-only centered unwhitened PCA up to32; numerical rank singular values > max(s0*1e-8,1e-12)','ridge':'One-hot three-class regression; tune MSE picks one alpha, fixed ascending tie-break; no class standardization','alphas':ALPHAS,'class_tie_tolerance':TIE,'class_rule':'Lowest index within1e-10 of max, applied universally; raw argmax also saved to audit floating ties','scene_split':[24,8,8],'labels':'Each factor uses only low/baseline/high, each scene balanced3. Baseline reused across tasks, not additional trajectories.','state_input':'Only x,y,z,vx,vy,vz; excludes parameter labels/dynamics info/force/contacts','scope':'Supervised factor-conditional diagnostic; F/m and contact confounds remain; not unsupervised or unique physical identification','bootstrap':'5000 whole-dev-scene resamples, 8 scenes, conditional and no multiplicity correction'};dump(O/'config.json',cfg);dump(O/'t0_audit.json',t0);rows=[]
 for source,A in arrays.items():
  for version in VERSIONS:
   for factor in FACTORS:
    X,y,scene,split,sourceix=build(m,A,lookup,factor,version);masks=[split==i for i in range(3)];xx=[X[b] for b in masks];yy=[np.eye(3)[y[b]] for b in masks];mu=xx[0].mean(0);u,s,vh=la.svd(xx[0]-mu,full_matrices=False);rank=min(32,int(np.sum(s>max(s[0]*1e-8,1e-12))));basis=vh[:rank].T;Z=[(x-mu)@basis for x in xx];my=yy[0].mean(0);G=Z[0].T@Z[0]/len(Z[0]);rhs=Z[0].T@(yy[0]-my)/len(Z[0]);weights=[la.solve(G+alpha*np.eye(rank),rhs,assume_a='pos') if rank else np.zeros((0,3)) for alpha in ALPHAS];loss=np.array([np.mean((Z[1]@w+my-yy[1])**2) for w in weights]);chosen=int(np.argmin(loss));coef=weights[chosen];scores=Z[2]@coef+my;p=predict_class(scores);ym=y[masks[2]];ss=scene[masks[2]];met=metrics(ym,p,ss);raw=metrics(ym,scores.argmax(1),ss);key=f'{source}_{version}_{factor}';np.savez_compressed(O/f'{key}.npz',prediction=p,scores=scores,truth=ym,scene=ss,source_indices=sourceix[masks[2]],pca_mean=mu,pca_basis=basis,singular_values=s,coefficients=coef,target_mean=my,alpha=ALPHAS[chosen],tune_losses=loss,raw_argmax=scores.argmax(1));row={'source':source,'version':version,'factor':factor,'rank':rank,'t0_exact_unique_scene_rank_bound':23 if version=='t0' else None,'t0_rank_exceeds_scene_bound':bool(rank>23) if version=='t0' else None,'original_dimensions':X.shape[1],'alpha':ALPHAS[chosen],'tune_onehot_mse':float(loss[chosen]),'file':key+'.npz',**met,'raw_argmax_balanced_accuracy':raw['balanced_accuracy'],'max_score_margin':float(np.max(np.sort(scores,1)[:,-1]-np.sort(scores,1)[:,-2]))}
    if version=='t0':assert abs(met['balanced_accuracy']-1/3)<1e-12 and all(abs(e['accuracy']-1/3)<1e-12 for e in met['per_scene']),row
    acc=np.array([x['accuracy'] for x in met['per_scene']]);rng=np.random.default_rng(571);ix=rng.integers(8,size=(5000,8));row['scene_bootstrap_ci95']=np.percentile(acc[ix].mean(1),[2.5,97.5]).tolist();rows.append(row)
   print(source,version,'complete',flush=True)
   dump(O/'results.json',rows)
 dump(O/'COMPUTATION_COMPLETE.json',{'files':len(rows),'status':'computed; independent verification pending'})

def verify(data,features,O):
 m,arrays,lookup,t0=load(data,features);rows=json.loads((O/'results.json').read_text());maxerr=0.;reconerr=0.;checks=[]
 for row in rows:
  z=np.load(O/row['file']);X,y,scene,split,ix=build(m,arrays[row['source']],lookup,row['factor'],row['version']);ms=[split==i for i in range(3)];P=z['pca_basis'];mu=X[ms[0]].mean(0);A=[(X[b]-mu)@P for b in ms];Y=[np.eye(3)[y[b]] for b in ms];my=Y[0].mean(0);u,s,v=la.svd(A[0],full_matrices=False);rhs=u.T@(Y[0]-my);ws=[v.T@((s/(s*s+len(A[0])*a))[:,None]*rhs) for a in ALPHAS];loss=np.array([np.mean((A[1]@w+my-Y[1])**2) for w in ws]);best=int(loss.argmin());assert ALPHAS[best]==row['alpha'];scores=A[2]@ws[best]+my;maxerr=max(maxerr,float(np.max(np.abs(scores-z['scores']))));pred=predict_class(scores);assert np.array_equal(pred,z['prediction']) and np.array_equal(y[ms[2]],z['truth']) and np.array_equal(scene[ms[2]],z['scene']) and np.array_equal(ix[ms[2]],z['source_indices']);met=metrics(y[ms[2]],pred,scene[ms[2]]);assert met['confusion']==row['confusion'] and met['balanced_accuracy']==row['balanced_accuracy'];r=P.shape[1]
  if r:
   centered=X[ms[0]]-mu;covP=centered.T@(centered@P);expected=P*(z['singular_values'][:r]**2);err=float(np.max(np.abs(covP-expected))/max(1.,np.max(np.abs(expected))));reconerr=max(reconerr,err);assert err<1e-7
  checks.append({'file':row['file'],'balanced_accuracy':met['balanced_accuracy'],'passed':True})
 assert maxerr<1e-7
 out={'status':'passed','files':len(rows),'score_max_error':maxerr,'pca_eigen_equation_relative_max_error':reconerr,'source_truth_scene_alpha_predictions_confusion':'exact or recomputed','method':'Independent thin SVD ridge versus primary normal-equation solve','checks':checks,'code_sha256':sha(Path(__file__))};dump(O/'independent_validation.json',out);dump(O/'COMPLETE.json',{'status':'passed','files':len(rows)});print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args();(verify if a.verify else run)(a.data,a.features,a.output)
