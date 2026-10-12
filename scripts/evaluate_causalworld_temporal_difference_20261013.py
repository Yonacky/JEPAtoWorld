"""G5c frozen exploratory temporal differences on previously inspected dev scenes."""
import argparse,json
from pathlib import Path
import evaluate_causalworld_dynamics_20261015 as base
import numpy as np
VERSIONS=['delta30','delta120','concat_delta30_delta120','adjacent8','adjacent8_per_second']
ORIGINAL_LOAD=base.load

def load(data,features):
 m,arrays,lookup,t0=ORIGINAL_LOAD(data,features)
 with np.load(data) as z:m['time_seconds']=z['time_seconds'].astype(float)
 for ix in lookup.values():
  t=m['time_seconds'][ix];assert np.all(np.diff(t)>0);np.testing.assert_allclose(t,np.array(base.TIMES)*.004,rtol=0,atol=1e-14)
 for a in arrays.values():assert np.count_nonzero(a-a)==0
 return m,arrays,lookup,t0

def build(m,A,lookup,factor,version):
 X=[];y=[];scene=[];split=[];source=[]
 for s in range(40):
  for lev in range(3):
   ix=lookup[s,'baseline',1] if lev==1 else lookup[s,factor,lev];v=A[ix];d=np.diff(v,axis=0)
   if version=='delta30':x=v[5]-v[0]
   elif version=='delta120':x=v[8]-v[0]
   elif version=='concat_delta30_delta120':x=np.concatenate([v[5]-v[0],v[8]-v[0]])
   elif version=='adjacent8':x=d.ravel()
   elif version=='adjacent8_per_second':x=(d/np.diff(m['time_seconds'][ix])[:,None]).ravel()
   else:raise ValueError(version)
   X.append(x);y.append(lev);scene.append(s);split.append(m['split'][ix[0]]);source.append(ix)
 return tuple(map(np.asarray,[X,y,scene,split,source]))

def independent_build(m,A,lookup,factor,version):
 # Reconstruct each row by explicit source-index arithmetic, without np.diff.
 result=build(m,A,lookup,factor,version);X,y,scene,split,indices=result;other=[]
 for ix in indices:
  if version=='delta30':pairs=[(5,0)]
  elif version=='delta120':pairs=[(8,0)]
  elif version=='concat_delta30_delta120':pairs=[(5,0),(8,0)]
  else:pairs=[(k,k-1) for k in range(1,9)]
  pieces=[]
  for end,start in pairs:
   value=A[ix[end]]-A[ix[start]]
   if version=='adjacent8_per_second':value=value/(m['time_seconds'][ix[end]]-m['time_seconds'][ix[start]])
   pieces.append(value)
  other.append(np.concatenate(pieces))
 other=np.asarray(other);np.testing.assert_array_equal(X,other)
 return other,y,scene,split,indices

def comparisons(O,B,verify=False):
 rows=json.loads((O/'results.json').read_text());old=json.loads((B/'results.json').read_text());out=[]
 for r in rows:
  z=np.load(O/r['file']);acc=np.array([np.mean(z['prediction'][z['scene']==s]==z['truth'][z['scene']==s]) for s in np.unique(z['scene'])])
  for b in old:
   if (b['source'],b['factor'])!=(r['source'],r['factor']):continue
   p=np.load(B/b['file']);assert np.array_equal(z['scene'],p['scene']) and np.array_equal(z['truth'],p['truth']) and np.array_equal(z['source_indices'],p['source_indices'])
   prior=np.array([np.mean(p['prediction'][p['scene']==s]==p['truth'][p['scene']==s]) for s in np.unique(p['scene'])]);difference=acc-prior;rng=np.random.default_rng(571)
   sampled=rng.choice(difference,size=(5000,8),replace=True) if verify else difference[rng.integers(8,size=(5000,8))]
   out.append({'source':r['source'],'factor':r['factor'],'version':r['version'],'baseline':b['version'],'delta_balanced_accuracy':float(difference.mean()),'paired_scene_ci95':np.percentile(sampled.mean(1),[2.5,97.5]).tolist(),'per_scene_difference':difference.tolist()})
 if verify:
  assert out==json.loads((O/'paired_comparisons.json').read_text())
  for r in rows:
   z=np.load(O/r['file']);acc=np.array([np.mean(z['prediction'][z['scene']==s]==z['truth'][z['scene']==s]) for s in np.unique(z['scene'])]);rng=np.random.default_rng(571);np.testing.assert_array_equal(np.percentile(rng.choice(acc,(5000,8)).mean(1),[2.5,97.5]),r['scene_bootstrap_ci95'])
 else:base.dump(O/'paired_comparisons.json',out)
 return len(out)

def main(a):
 base.load=load;base.build=independent_build if a.verify else build;base.VERSIONS=VERSIONS
 if a.verify:
  base.verify(a.data,a.features,a.output);n=comparisons(a.output,a.baseline,True);v=json.loads((a.output/'independent_validation.json').read_text());v.update({'paired_comparisons_recomputed':n,'bootstrap_recomputed':45,'temporal_input_independent_index_arithmetic':'exact','wrapper_sha256':base.sha(Path(__file__))});base.dump(a.output/'independent_validation.json',v)
 else:
  base.run(a.data,a.features,a.output);comparisons(a.output,a.baseline)
  p=a.output/'config.json';cfg=json.loads(p.read_text());cfg.update({'scope':'Exploration on previously inspected dev; supervised conditional diagnostic, not fresh confirmation; F/m remains confounded','wrapper_sha256':base.sha(Path(__file__)),'dependency_sha256':{str(Path(base.__file__)):base.sha(Path(base.__file__))},'baseline_manifest_sha256':base.sha(a.baseline/'manifest.json'),'delta0_selfcheck':'all input arrays minus themselves are exactly zero; balanced prior classifier gives1/3','temporal_intervals_seconds':(np.diff(base.TIMES)*.004).tolist(),'paired_comparisons':'All five G5b baselines for every configuration;5000 paired scene resamples seed571; exploratory and uncorrected','primary_comparators':{'delta30':'t30','delta120':'t120','concat_delta30_delta120':'concat_0_30_120','adjacent8':'concat_0_30_120','adjacent8_per_second':'concat_0_30_120'},'capacity_limitation':'adjacent inputs use9 frames vs3-frame baseline; per-second reweighting changes PCA energy and ridge scales, not a controlled derivative-only comparison'});base.dump(p,cfg)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--verify',action='store_true');main(p.parse_args())
