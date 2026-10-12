"""Independent G5i audit: raw feature distances, exact ratio membership, scene bootstrap."""
from pathlib import Path
import argparse,json,math,hashlib
import numpy as np
TIMES=(0,1,5,10,20,30,60,90,120)
MASS=(1,2,4)
FORCE=(1,2,3)
PAIRS=(((0,0),(1,1)),((1,0),(2,1)))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normdiff(a,b):return math.sqrt(math.fsum(float(x)*float(x) for x in a-b))
def close(a,b):
 if a is None or b is None:assert a is None and b is None,(a,b);return 0.
 np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11);return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))

def main(a):
 z=np.load(a.data);f=np.load(a.features);fm=np.load(a.features.parent/'sample_manifest.npz');primary=json.loads(a.results.read_text(encoding='utf-8'));N=len(z['scene']);assert N==6480
 for key in ['split','episode','frame','labels']:np.testing.assert_array_equal(z[key],fm[key])
 index={}
 for i in range(N):
  sc=int(z['scene'][i]);mode=str(z['contact_mode'][i]);m=int(z['mass_level'][i]);force=int(z['force_level'][i]);t=int(z['frame'][i]);key=(sc,mode,m,force,t);assert key not in index;index[key]=i;assert sc in range(40) and mode in ['on','off'] and m in range(3) and force in range(3) and t in TIMES;assert int(z['split'][i])==(0 if sc<24 else 1 if sc<32 else 2)
 for sc in range(40):
  for mode in ['on','off']:
   for m in range(3):
    for force in range(3):
     ids=[index[sc,mode,m,force,t] for t in TIMES];assert len(set(z['episode'][ids]))==1
 rows={};checks=[];maxerr=0.;tiny=[];source_maps={}
 for source in ['mean','grid']:
  A=f[source].astype(np.float64);assert A.shape==(N,1280 if source=='mean' else 5120) and np.isfinite(A).all()
  for sc in range(40):
   initial=[index[sc,mode,m,force,0] for mode in ['on','off'] for m in range(3) for force in range(3)];np.testing.assert_array_equal(A[initial],np.repeat(A[initial[:1]],18,axis=0));checks.append({'source':source,'scene':sc,'maxabs':0.})
   for mode in ['on','off']:
    for version in ['t0','t30','t120','delta30_delta120']:
     vals={}
     for m in range(3):
      for force in range(3):
       key=sc,mode,m,force
       if version=='delta30_delta120':vals[m,force]=np.concatenate((A[index[key+(30,)]]-A[index[key+(0,)]],A[index[key+(120,)]]-A[index[key+(0,)]]))
       else:vals[m,force]=A[index[key+(int(version[1:]),)]]
     for pi,(anchor,target) in enumerate(PAIRS):
      # Integer cross multiplication avoids reusing floating-ratio membership logic.
      eligible=[(m,force) for m in range(3) for force in range(3) if FORCE[force]*MASS[anchor[0]]!=FORCE[anchor[1]]*MASS[m]];assert len(eligible)==7
      ds=[normdiff(vals[anchor],vals[c]) for c in eligible];den=sorted(ds)[3];num=normdiff(vals[anchor],vals[target]);value=num/den if den>0 else None;key=source,version,sc,mode,pi;assert key not in rows;rows[key]={'num':num,'den':den,'ratio':value,'comparison':ds}
      if 0<den<=1e-14:tiny.append(list(key)+[den])
      source_maps['|'.join(map(str,key))]={'anchor_rows':[index[(sc,mode,*anchor,t)] for t in TIMES],'other_rows':[index[(sc,mode,*target,t)] for t in TIMES],'unequal_cells':eligible}
 assert len(primary['rows'])==len(rows)==1280
 for q in primary['rows']:
  r=rows[q['source'],q['version'],q['scene'],q['mode'],q['pair']]
  for aa,bb in [('equal_ratio_distance','num'),('unequal_ratio_median_distance','den'),('normalized_distance','ratio'),('comparison_distances','comparison')]:maxerr=max(maxerr,close(q[aa],r[bb]))
 assert sorted(checks,key=lambda x:(x['source'],x['scene']))==sorted(primary['t0_checks'],key=lambda x:(x['source'],x['scene']))
 summaries=[]
 for source in ['mean','grid']:
  for version in ['t0','t30','t120','delta30_delta120']:
   v={mode:[None if any(rows[source,version,sc,mode,pi]['ratio'] is None for pi in range(2)) else math.fsum(rows[source,version,sc,mode,pi]['ratio'] for pi in range(2))/2 for sc in range(32,40)] for mode in ['on','off']};d=None;ci=None
   if all(x is not None for vals in v.values() for x in vals):
    paired=np.array([v['on'][j]-v['off'][j] for j in range(8)]);d=float(paired.mean());rng=np.random.default_rng(571);ci=np.percentile(rng.choice(paired,size=(5000,8),replace=True).mean(1),[2.5,97.5]).tolist()
   q=next(x for x in primary['summary'] if x['source']==source and x['version']==version)
   for mode in ['on','off']:
    for aa,bb in zip(v[mode],q['dev_scene_values'][mode]):maxerr=max(maxerr,close(aa,bb))
   maxerr=max(maxerr,close(d,q['ON_minus_OFF_mean']),close(ci,q['paired_scene_ci95']));summaries.append({'source':source,'version':version,'dev_scene_values':v,'ON_minus_OFF_mean':d,'paired_scene_ci95':ci})
 images=z['images'];state=z['actual_states'];assert len(primary['image_checks'])==160
 for q in primary['image_checks']:
  aa,bb=PAIRS[q['pair']];ia=[index[q['scene'],q['mode'],*aa,t] for t in TIMES];ib=[index[q['scene'],q['mode'],*bb,t] for t in TIMES];counts=[int(np.count_nonzero(np.max(images[i]!=images[j],axis=-1))) for i,j in zip(ia,ib)];assert counts==q['different_pixels_by_frame'];assert (sum(counts)==0)==q['equal_pixels_all9'];maxerr=max(maxerr,close(float(np.max(np.abs(state[ia]-state[ib]))),q['sampled_tool_state_maxabs']))
 result={'status':'PASS','distance_rows':1280,'image_state_pairs':160,'summary_count':8,'maximum_absolute_numeric_difference':maxerr,'tiny_positive_denominators':tiny,'NA_rule':'den==0 only; positive tiny cases separately reported','method':'Raw features; math.fsum Euclidean norms; exact integer cross-product unequal-ratio membership; sorted seven-distance median; rng.choice paired scene bootstrap','input_sha256':{str(p):sha(p) for p in [a.data,a.features,a.features.parent/'sample_manifest.npz',a.results,Path(__file__)]},'independent_summary':summaries}
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(a.output/'source_pair_maps.json').write_text(json.dumps(source_maps,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','independent_summary']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
