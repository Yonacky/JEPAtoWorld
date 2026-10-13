"""G7e image-family fork of frozen G5l; one16/8 split, no parameter changes."""
import contact_ratio_readout_20261013 as b
import family_provenance as provenance
from pathlib import Path
import argparse,json
import numpy as np
from scipy import linalg as la

def main(a):
 O=a.output;O.mkdir(parents=True,exist_ok=False);mp=np.load(a.directions/'mapping.npz');data=np.load(a.data);audit=provenance.verify_manifest(a.directions/'manifest.json',a.path_map);b.dump(O/'direction_provenance_audit.json',audit)
 baseline_audit=provenance.verify_manifest(a.baseline/'manifest.json',a.path_map,include_config=True);b.dump(O/'baseline_provenance_audit.json',baseline_audit)
 assert any(Path(c['local_path']).resolve()==a.data.resolve() and c['sha256']==b.sha(a.data) for c in baseline_audit['checks'])
 assert json.loads((a.baseline/'config.json').read_text())['input_kind']=='image_g5j'
 rows=[];audits=[];checks=[];maps={};maxerr=0.
 for mode in ['on','off']:
  X=mp[mode+'_pca_scores'];sc=mp[mode+'_scene'];ix=mp[mode+'_source_indices'];y=np.log(data['labels'][ix[:,0],2]/data['labels'][ix[:,0],0]);baseline=np.load(a.baseline/(mode+'_logratio.npz'));sd=float(baseline['target_std']);fit=sc<16;probe=(sc>=16)&(sc<24);tu=(sc>=24)&(sc<32);de=sc>=32;assert [x.sum() for x in [fit,probe,tu,de]]==[144,72,72,72]
  mu=X[fit].mean(0);ym=y[fit].mean();xc=X[fit]-mu;yc=(y[fit]-ym)/sd;C=xc.T@xc/(fit.sum()-1);c=xc.T@yc/(fit.sum()-1);ev,V=la.eigh(C);mask=ev>ev[-1]*32*np.finfo(float).eps;W=(V*np.where(mask,1/np.sqrt(np.maximum(ev,np.finfo(float).tiny)),0))@V.T;Wi=(V*np.sqrt(np.maximum(ev,0)))@V.T;wc=W@c
  if la.norm(wc)==0:E=np.eye(32);direct=E
  else:
   u=wc/la.norm(wc);E=np.eye(32)-np.outer(Wi@u,u@W);ci=(V*np.where(mask,1/np.maximum(ev,np.finfo(float).tiny),0))@V.T@c;direct=np.eye(32)-np.outer(c,ci)/(c@ci)
  maperr=float(np.max(abs(E-direct)));assert maperr<1e-9;idem=float(np.max(abs(E@E-E)));cross=float(la.norm(E@c)/la.norm(c));assert idem<1e-10 and cross<1e-10;erased=(X-mu)@E.T+mu;maps[mode+'_E']=E;maps[mode+'_mean']=mu;maps[mode+'_covariance']=C;maps[mode+'_crosscovariance']=c;maps[mode+'_source_indices']=ix;rank=int(np.linalg.matrix_rank(E));audit={'mode':mode,'covariance_rank':int(mask.sum()),'transform_rank':rank,'independent_formula_max_error':maperr,'idempotence_max_error':idem,'fit_crosscovariance_ratio':cross,'splits':{}}
  for name,sel in [('fit16',fit),('probe8',probe),('tune8',tu),('dev8',de)]:
   ys=(y[sel]-y[sel].mean())/sd;co=(X[sel]-X[sel].mean(0)).T@ys/(sel.sum()-1);ce=(erased[sel]-erased[sel].mean(0)).T@ys/(sel.sum()-1);audit['splits'][name]={'raw_crosscovariance_norm':float(la.norm(co)),'erased_crosscovariance_norm':float(la.norm(ce)),'crosscovariance_norm_ratio':float(la.norm(ce)/la.norm(co)),'relative_squared_damage':float(np.sum((erased[sel]-X[sel])**2)/np.sum((X[sel]-mu)**2))}
  audits.append(audit)
  for method,Z in [('raw32',X),('leace32',erased)]:
   mx=Z[probe].mean(0);my=y[probe].mean();zs=[Z[q]-mx for q in [probe,tu,de]];target=(y[probe]-my)/sd;G=zs[0].T@zs[0]/probe.sum();rhs=zs[0].T@target/probe.sum();ws=[la.solve(G+alpha*np.eye(32),rhs,assume_a='pos') for alpha in b.ALPHAS];loss=np.array([np.mean((zs[1]@w-(y[tu]-my)/sd)**2) for w in ws]);best=int(loss.argmin());w=ws[best];pred=zs[2]@w*sd+my
   U,s,Vh=la.svd(zs[0],full_matrices=False);iw=[Vh.T@((s/(s*s+probe.sum()*alpha))*(U.T@target)) for alpha in b.ALPHAS];il=np.array([np.mean((zs[1]@ww-(y[tu]-my)/sd)**2) for ww in iw]);assert il.argmin()==best;err=float(np.max(abs(pred-(zs[2]@iw[best]*sd+my))));assert err<1e-9;maxerr=max(err,maxerr);key=mode+'_'+method;np.savez_compressed(O/(key+'.npz'),truth=y[de],prediction=pred,scene=sc[de],source_indices=ix[de],coef=w,probe_feature_mean=mx,probe_target_mean=my,target_std=sd,alpha=b.ALPHAS[best],tune_losses=loss)
   rows.append({'mode':mode,'method':method,'alpha':b.ALPHAS[best],**b.metrics(y[de],pred,sc[de],sd)});checks.append({'model':key,'svd_prediction_max_error':err});print(key,rows[-1]['r2'],flush=True)
  rr=rows[-2:];delta=np.array(list(rr[1]['scene_log_mse'].values()))-np.array(list(rr[0]['scene_log_mse'].values()));rng=np.random.default_rng(571);audit['leace_minus_raw_logMSE']=float(delta.mean());audit['paired_scene_ci95']=np.percentile(rng.choice(delta,(5000,8)).mean(1),[2.5,97.5]).tolist()
 np.savez_compressed(O/'transforms.npz',**maps);b.dump(O/'results.json',{'models':rows,'audits':audits});b.dump(O/'independent_validation.json',{'status':'PASS','models':4,'max_prediction_error':maxerr,'checks':checks,'formula':'Whiten-project-unwhiten vs scalar c(Cinv c)^T/(c^T Cinv c)','probe':'SVD vs normal equations'});b.dump(O/'config.json',{'eraser_scenes':list(range(16)),'probe_scenes':list(range(16,24)),'tune_scenes':list(range(24,32)),'dev_scenes':list(range(32,40)),'PCA':'Historical unlabeled24train PCA32; internal transductive split only, no dev','formula':'float64 empirical covariance, no shrinkage, no trace constraint, machine-epsilon pseudoinverse threshold','alphas':b.ALPHAS,'scope':'Supervised concept erasure, old-dev exploratory; 8scene probe, not unique physical factor','sources':{'paper':'https://arxiv.org/html/2306.03819v4#S4.SS3','code':'https://raw.githubusercontent.com/EleutherAI/concept-erasure/main/concept_erasure/leace.py'},'hashes':{str(p):b.sha(p) for p in [a.data,a.directions/'manifest.json',a.baseline/'manifest.json',Path(__file__),Path(b.__file__),Path('docs/leace_observable_protocol_20261013.txt')]}})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--directions',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--path-map',type=Path);a=p.parse_args();main(a);provenance.finalize(a.output,[Path(__file__),Path(b.__file__),Path(provenance.__file__)])
