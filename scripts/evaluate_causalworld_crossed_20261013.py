"""G5d-eval frozen 16-model crossed mass-force supervised diagnostic."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
import json,hashlib,argparse
import numpy as np
from scipy import linalg as la
ALPHAS=[1e-5,.001,.1,1.,10.];TIMES=[0,1,5,10,20,30,60,90,120];TIE=1e-10

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
def cls(scores):return np.argmax(scores>=scores.max(1,keepdims=True)-TIE,axis=1)
def metric(y,p,scene):
 cm=np.zeros((3,3),int);np.add.at(cm,(y,p),1);assert (cm.sum(1)>0).all();a=np.array([np.mean(y[scene==s]==p[scene==s]) for s in np.unique(scene)]);rng=np.random.default_rng(20261016);ii=rng.integers(len(a),size=(5000,len(a)));return {'balanced_accuracy':float(np.mean(np.diag(cm)/cm.sum(1))),'confusion':cm.tolist(),'scene_accuracy':{str(s):float(v) for s,v in zip(np.unique(scene),a)},'scene_bootstrap_ci95':np.percentile(a[ii].mean(1),[2.5,97.5]).tolist(),'samples':len(y)}
def load(data,features):
 z=np.load(data);f=np.load(features);fm=np.load(features.parent/'sample_manifest.npz');m={k:z[k] for k in ['split','scene','episode','frame','mass_level','force_level','actual_states','state_names','labels']}
 for k in ['split','episode','frame','labels']:assert np.array_equal(fm[k],m[k]),k
 assert list(m['state_names'])==['x','y','z','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz'];assert m['actual_states'].shape==(3240,1,13);grid=f['grid'].astype(float);assert grid.shape==(3240,5120)
 arrays={'grid':grid,'state_posvel':m['actual_states'][:,0,[0,1,2,7,8,9]].astype(float)};assert all(np.isfinite(x).all() for x in arrays.values());lookup={};t0=[]
 for s in range(40):
  q=np.flatnonzero(m['scene']==s);assert len(q)==81 and np.all(m['split'][q]==(0 if s<24 else 1 if s<32 else 2))
  for mi in range(3):
   for fi in range(3):
    ix=q[(m['mass_level'][q]==mi)&(m['force_level'][q]==fi)];ix=ix[np.argsort(m['frame'][ix])];assert np.array_equal(m['frame'][ix],TIMES);assert len(np.unique(m['episode'][ix]))==1;assert np.allclose(m['labels'][ix],[ [.01,.02,.04][mi],.175,[.01,.02,.03][fi] ],rtol=0,atol=1e-12);lookup[s,mi,fi]=ix
  zero=q[m['frame'][q]==0];assert np.array_equal(z['images'][zero],np.repeat(z['images'][zero[:1]],9,axis=0))
  for source,A in arrays.items():
   diff=float(np.max(np.abs(A[zero]-A[zero[0]])));scale=max(1.,float(np.max(np.abs(A[zero]))));assert diff<=2e-6*scale;t0.append({'scene':s,'source':source,'max_abs':diff,'scale':scale})
 return m,arrays,lookup,t0

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();O=a.output;O.mkdir(parents=True,exist_ok=True);m,arrays,lookup,t0=load(a.data,a.features)
 dump(O/'config.json',{'scope':'Supervised condition decoding and transfer to unseen nuisance middle level (interpolation), not target-class zero-shot, unique F/m identification or independent factors','inputs':{str(p):sha(p) for p in [a.data,a.features,a.features.parent/'sample_manifest.npz']},'source_sha256':sha(Path(__file__)),'split_scenes':[24,8,8],'parameters':{'mass':[.01,.02,.04],'force':[.01,.02,.03],'friction':.175},'alphas':ALPHAS,'rank':32,'PCA':'train-only centered unwhitened, retain min32,numerical rank; state6/12 naturally lower rank','versions':['t120','delta30_delta120'],'regimes':['ordinary_all9','held_nuisance_middle'],'selection':'onehot tune MSE, only eligible nuisance low/high in held regime; fixed five-alpha ascending tie break','class_tie_tolerance':TIE,'capacity':'ordinary216train+72tune vs held144train+48tune; gap confounds training count and nuisance coverage. No extra count-matched models.','state_input':'x,y,z,vx,vy,vz only, excludes parameters and contacts','temporal_definition':'deltas from common t0 to30 and120, concatenate; force only steps1..30, dt.004','bootstrap':'5000 paired whole-dev-scene resamples;8 scenes, conditional, targets dependent and not independent confirmations'});dump(O/'t0_audit.json',t0);results=[];validation=[]
 for source,A in arrays.items():
  for version in ['t120','delta30_delta120']:
   X=[];sc=[];spl=[];ml=[];fl=[];sourceix=[]
   for s in range(40):
    for mi in range(3):
     for fi in range(3):
      ix=lookup[s,mi,fi];v=A[ix];X.append(v[8] if version=='t120' else np.r_[v[5]-v[0],v[8]-v[0]]);sc.append(s);spl.append(int(m['split'][ix[0]]));ml.append(mi);fl.append(fi);sourceix.append(ix)
   X=np.array(X);sc=np.array(sc);spl=np.array(spl);ml=np.array(ml);fl=np.array(fl);sourceix=np.array(sourceix)
   for target in ['mass','force']:
    y=ml if target=='mass' else fl;nuis=fl if target=='mass' else ml
    for regime in ['ordinary_all9','held_nuisance_middle']:
     allowed=np.ones(len(y),bool) if regime=='ordinary_all9' else nuis!=1;train=(spl==0)&allowed;tune=(spl==1)&allowed;dev=spl==2;assert not np.intersect1d(sc[train],sc[dev]).size;assert set(y[train])=={0,1,2}
     if regime.startswith('held'):assert np.all(nuis[train]!=1) and np.all(nuis[tune]!=1)
     mu=X[train].mean(0);xc=X[train]-mu;u,s,vh=la.svd(xc,full_matrices=False);rank=min(32,int(np.sum(s>max(s[0]*1e-8,1e-12))));P=vh[:rank].T;Z=[(X[q]-mu)@P for q in [train,tune,dev]];YY=[np.eye(3)[y[q]] for q in [train,tune,dev]];ym=YY[0].mean(0);g=Z[0].T@Z[0]/sum(train);c=Z[0].T@(YY[0]-ym)/sum(train);ws=[la.solve(g+alpha*np.eye(rank),c,assume_a='pos') if rank else np.zeros((0,3)) for alpha in ALPHAS];loss=np.array([np.mean((Z[1]@w+ym-YY[1])**2) for w in ws]);best=int(loss.argmin());scores=Z[2]@ws[best]+ym;pred=cls(scores);key=f'{source}_{version}_{target}_{regime}';np.savez_compressed(O/(key+'.npz'),scores=scores,prediction=pred,truth=y[dev],scene=sc[dev],nuisance=nuis[dev],source_indices=sourceix[dev],train_source_indices=sourceix[train],tune_source_indices=sourceix[tune],pca_mean=mu,pca_basis=P,singular_values=s,coefficients=ws[best],target_mean=ym,alpha=ALPHAS[best],tune_losses=loss)
     # Independent SVD ridge, reselect on eligible tune only, independently score saved predictions.
     uu,ss,vv=la.svd(Z[0],full_matrices=False);rhs=uu.T@(YY[0]-ym);ww=[vv.T@((ss/(ss*ss+len(Z[0])*alpha))[:,None]*rhs) for alpha in ALPHAS];ll=np.array([np.mean((Z[1]@w+ym-YY[1])**2) for w in ww]);assert int(ll.argmin())==best;rec=Z[2]@ww[best]+ym;error=float(np.max(np.abs(rec-scores)));assert error<1e-7 and np.array_equal(cls(rec),pred)
     pack=np.load(O/(key+'.npz'));assert np.array_equal(pack['source_indices'],sourceix[dev]);cov=xc.T@(xc@P);eigerr=float(np.max(np.abs(cov-P*s[:rank]**2))/max(1.,np.max(np.abs(P*s[:rank]**2)))) if rank else 0.;assert eigerr<1e-7
     row={'source':source,'version':version,'target':target,'regime':regime,'file':key+'.npz','rank':rank,'width':X.shape[1],'train_samples':int(sum(train)),'tune_samples':int(sum(tune)),'alpha':ALPHAS[best],'tune_mse':float(loss[best])}
     for label,mask in [('all9',np.ones(sum(dev),bool)),('nuisance_lowhigh',nuis[dev]!=1),('nuisance_middle',nuis[dev]==1)]:
      row[label]=metric(y[dev][mask],pred[mask],sc[dev][mask]);ind=metric(pack['truth'][mask],pack['prediction'][mask],pack['scene'][mask]);assert ind['confusion']==row[label]['confusion']
     results.append(row);validation.append({'file':key,'independent_score_max_error':error,'pca_eigen_equation_error':eigerr,'eligible_tune_alpha_and_classes':'exact'});print(key,'middle',row['nuisance_middle']['balanced_accuracy'],flush=True)
 dump(O/'results.json',results);dump(O/'independent_validation.json',{'status':'passed','models':len(results),'checks':validation});print('DONE',flush=True)
if __name__=='__main__':main()
