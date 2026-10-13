"""Independent G7i audit: explicit tuple enumeration, coordinate mask/solve,
full-space first7 per scene/task/model, ALL saved edit reductions and bootstrap.
Does not import evaluator functions or fit any model.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import argparse,json,hashlib,itertools
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);a=p.parse_args();O=a.evaluation
 run=json.loads((O/'run_config.json').read_text());args=run['arguments'];cfg=run['frozen_config']
 for path,h in run['source_sha256'].items():assert sha(path)==h,path
 z=np.load(args['data']);x=np.load(args['features'])['grid'].astype(float);m=np.load(args['models']);U=m['U'];initial=np.load(args['joint0'])['joint_0']
 result=json.loads((O/'results.json').read_text());names=result['models'];assert set(names)=={'no_edit','joint0_twoanchors','orth_lambda1','oblique_lambda1'}
 pairs=json.loads((O/'pair_map.json').read_text());scenes=[p['recipient'] for p in pairs];half=len(scenes)//2
 assert len(set(scenes))==2*half
 assert all(pairs[i]['donor']==scenes[(i+half)%(2*half)] for i in range(2*half))
 assert len({frozenset([p['recipient'],p['donor']]) for p in pairs})==half
 saved=np.load(O/'scene_metrics.npz')['metrics'];records=np.load(O/'edit_records.npz');details=np.load(O/'edit_metrics.npz');samples=np.load(O/'prediction_samples.npz')
 computed=np.zeros_like(saved);maxerr=0.;maxprediction=0.;count=0;ksall=[(0,),(1,),(2,),(0,1),(0,2),(1,2)]
 codes=np.array(list(itertools.product(range(3),repeat=3)))
 for si,pair in enumerate(pairs):
  sc=pair['recipient'];donor=pair['donor'];ii=np.flatnonzero(z['scene']==sc);jj=np.flatnonzero(z['scene']==donor)
  np.testing.assert_array_equal(ii,pair['recipient_rows']);np.testing.assert_array_equal(jj,pair['donor_rows'])
  np.testing.assert_array_equal(z['codes'][ii],codes);np.testing.assert_array_equal(z['codes'][jj],codes)
  X=x[ii];Y=x[jj]
  for ti,ks in enumerate(ksall):
   other=[k for k in range(3) if k not in ks];rows=[]
   for source in range(27):
    for d in range(27):
     if not all(codes[source,k]!=codes[d,k] for k in ks):continue
     if not any(codes[source,k]!=codes[d,k] for k in other):continue
     tc=codes[source].copy();tc[list(ks)]=codes[d,list(ks)];target=int(tc@[9,3,1]);valid=int(source not in [0,26] and d not in [0,26] and target not in [0,26]);rows.append([source,d,target,valid])
   rows=np.array(rows);np.testing.assert_array_equal(rows,records[f's{sc}_t{ti}']);mask=rows[:,3].astype(bool)
   aa,dd,tt=rows[:7,:3].T
   v=(Y[dd]-X[aa]-(Y[[0,26]].mean(0)-X[[0,26]].mean(0)))@U
   for ni,name in enumerate(names):
    key=f'{name}_s{sc}_t{ti}';vals=details[key];assert vals.shape==(len(rows),5)
    computed[ni,si,ti,0]=vals[:,:4].mean(0);computed[ni,si,ti,1]=vals[mask,:4].mean(0)
    if name=='no_edit':delta=np.zeros_like(v)
    elif name=='joint0_twoanchors':delta=v@sum(initial[k] for k in ks)
    else:
     W=m[name];coordinates=v@W;keep=np.zeros(64,bool)
     for k in ks:keep[16*k:16*k+16]=True
     coordinates[:,~keep]=0;delta=np.linalg.solve(W.T,coordinates.T).T
    maxprediction=max(maxprediction,float(abs(delta-samples[key]).max()))
    edited=X[aa]+delta@U.T;dist=np.stack([np.sum((edited-X[j])**2,1) for j in range(27)],1)
    near=dist.argmin(1);den=np.sum((X[tt]-X[aa])**2,1);error=np.sum((edited-X[tt])**2,1)/den
    direct=np.column_stack([error,np.all(codes[near][:,list(ks)]==codes[dd][:,list(ks)],1),np.all(codes[near][:,other]==codes[aa][:,other],1),near==tt,near])
    np.testing.assert_allclose(direct,vals[:7],atol=1e-8);maxerr=max(maxerr,float(abs(direct-vals[:7]).max()));count+=7
 np.testing.assert_allclose(computed,saved,atol=1e-12)
 rng=np.random.default_rng(cfg['bootstrap_seed']);indices=rng.integers(0,half,(10000,half));ci_error=0.
 for ref in ['orth_lambda1','joint0_twoanchors']:
  diff=computed[names.index('oblique_lambda1')]-computed[names.index(ref)];d=(diff[:half]+diff[half:])/2
  primary=np.column_stack([d[:,:3,1,0].mean(1),d[:,1,1,1]])
  # Independent loop over pair draws, same frozen resamples.
  means=np.array([np.mean(primary[ix],axis=0) for ix in indices]);ci=np.quantile(means,[.00625,.99375],axis=0)
  expected=result['contrasts'][ref];np.testing.assert_allclose(ci,expected['primary_ci98_75'],atol=1e-12);np.testing.assert_allclose(primary.mean(0),expected['primary_difference'],atol=1e-12)
  ci_error=max(ci_error,float(abs(ci-np.array(expected['primary_ci98_75'])).max()))
 out=dict(status='PASS',independent_pairs=half,fullspace_sample_edits=count,fullspace_maxabs=maxerr,prediction_coordinate_solve_maxabs=maxprediction,all_saved_edit_reductions=True,bonferroni_quantiles=[.00625,.99375],ci_maxabs=ci_error,scope='fullspace first7 of EVERY scene/task/model; all saved edits aggregation, full tuple masks and pair bootstrap; no fitting',source_hashes=run['source_sha256'],audit_script_sha256=sha(__file__))
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
