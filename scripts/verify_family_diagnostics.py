"""Independent saved family diagnostic reconstruction, no model refitting."""
from pathlib import Path
import argparse,json,numpy as np

def main(a):
 d=np.load(a.data);f=np.load(a.features);A=f['grid'].astype(float);dr=json.loads((a.directions/'results.json').read_text());lr=json.loads((a.leace/'results.json').read_text());tr=np.load(a.leace/'transforms.npz');maxerr=0.;pcaerr=0.;count=0
 for mode in ['on','off']:
  p=np.load(a.baseline/(mode+'_pca.npz'));ix=p['source_indices'];sc=p['scene'];de=sc>=32;X=np.concatenate([A[ix[:,5]]-A[ix[:,0]],A[ix[:,8]]-A[ix[:,0]]],1);P=p['basis'];Z=(X-p['mean'])@P;y=np.log(d['labels'][ix[:,0],2]/d['labels'][ix[:,0],0]);train=X[sc<24]-p['mean'];expected=P*p['singular_values'][:32]**2;err=float(np.max(abs(train.T@(train@P)-expected))/max(1.,np.max(abs(expected))));assert err<1e-10;pcaerr=max(pcaerr,err)
  for r in [r for r in dr if r['mode']==mode]:
   q=np.load(a.directions/(mode+'_'+r['method']+'.npz'));coef=P@q['pca32_projection']@q['coefficients'];pred=(X[de]-p['mean'])@coef*q['target_std']+q['target_mean'];maxerr=max(maxerr,float(np.max(abs(pred-q['prediction']))));np.testing.assert_array_equal(y[de],q['truth']);assert abs(float(np.mean((pred-y[de])**2))-r['log_mse'])<1e-11;count+=1
   baseline=np.load(a.baseline/(mode+'_logratio.npz'));delta=np.array([np.mean((q['truth'][q['scene']==s]-q['prediction'][q['scene']==s])**2)-np.mean((q['truth'][q['scene']==s]-baseline['prediction'][q['scene']==s])**2) for s in range(32,40)]);rng=np.random.default_rng(571);np.testing.assert_allclose(np.percentile(rng.choice(delta,(5000,8)).mean(1),[2.5,97.5]),r['paired_scene_ci95'],atol=1e-13)
  E=tr[mode+'_E'];mu=tr[mode+'_mean'];erased=np.einsum('ij,nj->ni',E,Z-mu)+mu
  for r in [r for r in lr['models'] if r['mode']==mode]:
   q=np.load(a.leace/(mode+'_'+r['method']+'.npz'));zz=Z if r['method']=='raw32' else erased;pred=(zz[de]-q['probe_feature_mean'])@q['coef']*q['target_std']+q['probe_target_mean'];maxerr=max(maxerr,float(np.max(abs(pred-q['prediction']))));np.testing.assert_array_equal(y[de],q['truth']);assert abs(float(np.mean((pred-y[de])**2))-r['log_mse'])<1e-11;count+=1
  raw=np.load(a.leace/(mode+'_raw32.npz'));lea=np.load(a.leace/(mode+'_leace32.npz'));delta=np.array([np.mean((lea['truth'][lea['scene']==s]-lea['prediction'][lea['scene']==s])**2)-np.mean((raw['truth'][raw['scene']==s]-raw['prediction'][raw['scene']==s])**2) for s in range(32,40)]);rng=np.random.default_rng(571);audit=next(q for q in lr['audits'] if q['mode']==mode);np.testing.assert_allclose(np.percentile(delta[rng.integers(8,size=(5000,8))].mean(1),[2.5,97.5]),audit['paired_scene_ci95'],atol=1e-13)
 assert count==22 and maxerr<1e-10;out={'status':'PASS','models':count,'raw_reconstruction_max_error':maxerr,'pca_eigen_equation_max_relative_error':pcaerr,'paired_CIs_recomputed':20,'scope':'Saved parameter raw-feature reconstruction plus training PCA equation; model scripts independently use SVD ridge and scalar LEACE formula'};a.output.write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ['data','features','baseline','directions','leace','output']:p.add_argument('--'+k,type=Path,required=True)
 main(p.parse_args())
