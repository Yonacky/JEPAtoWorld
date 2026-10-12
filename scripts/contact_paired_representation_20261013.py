"""Fixed known-ratio pair distances in frozen image representations."""
from pathlib import Path
import numpy as np,json,hashlib,argparse
PAIRS=[((0,0),(1,1)),((1,0),(2,1))]
RATIOS=np.array([.01,.02,.03])[None,:]/np.array([.01,.02,.04])[:,None]
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 z=np.load(a.data);f=np.load(a.features);fm=np.load(a.features.parent/'sample_manifest.npz');meta={k:z[k] for k in ['scene','contact_mode','mass_level','force_level','frame','split','episode','labels']}
 for k in ['split','episode','frame','labels']:assert np.array_equal(meta[k],fm[k]),k
 lookup={};scenes=sorted(set(meta['scene'].tolist()));assert scenes==list(range(40));times=[0,1,5,10,20,30,60,90,120]
 for sc in scenes:
  for mode in ['on','off']:
   for m in range(3):
    for force in range(3):
     ix=np.flatnonzero((meta['scene']==sc)&(meta['contact_mode']==mode)&(meta['mass_level']==m)&(meta['force_level']==force));ix=ix[np.argsort(meta['frame'][ix])];assert np.array_equal(meta['frame'][ix],times);lookup[sc,mode,m,force]=ix
 rows=[];t0=[]
 for source in ['mean','grid']:
  A=f[source].astype(float)
  for sc in scenes:
   ix=np.flatnonzero((meta['scene']==sc)&(meta['frame']==0));diff=float(np.max(abs(A[ix]-A[ix[0]])));assert diff==0;t0.append({'source':source,'scene':sc,'maxabs':diff})
  for version in ['t0','t30','t120','delta30_delta120']:
   for sc in scenes:
    for mode in ['on','off']:
     values={}
     for m in range(3):
      for force in range(3):
       v=A[lookup[sc,mode,m,force]];values[m,force]=v[times.index(int(version[1:]))] if version.startswith('t') else np.r_[v[5]-v[0],v[8]-v[0]]
     for pairid,(anchor,other) in enumerate(PAIRS):
      dist=float(np.linalg.norm(values[anchor]-values[other]));comparison=[float(np.linalg.norm(values[anchor]-v)) for cell,v in values.items() if not np.isclose(RATIOS[cell],RATIOS[anchor],rtol=0,atol=1e-12)];den=float(np.median(comparison));rows.append({'source':source,'version':version,'scene':sc,'mode':mode,'pair':pairid,'equal_ratio_distance':dist,'unequal_ratio_median_distance':den,'normalized_distance':dist/den if den>0 else None,'comparison_distances':comparison})
 summary=[]
 for source in ['mean','grid']:
  for version in ['t0','t30','t120','delta30_delta120']:
   bymode={}
   for mode in ['on','off']:
    vals=[]
    for sc in range(32,40):
     q=[x['normalized_distance'] for x in rows if x['source']==source and x['version']==version and x['mode']==mode and x['scene']==sc]
     vals.append(float(np.mean(q)) if all(v is not None for v in q) else None)
    bymode[mode]=vals
   difference=None;ci=None
   if all(x is not None for v in bymode.values() for x in v):
    d=np.array(bymode['on'])-np.array(bymode['off']);difference=float(d.mean());rng=np.random.default_rng(571);ci=np.percentile(d[rng.integers(8,size=(5000,8))].mean(1),[2.5,97.5]).tolist()
   summary.append({'source':source,'version':version,'dev_scene_values':bymode,'ON_minus_OFF_mean':difference,'paired_scene_ci95':ci})
 image_checks=[];images=z['images'];states=z['actual_states']
 for sc in scenes:
  for mode in ['on','off']:
   for pairid,(aa,bb) in enumerate(PAIRS):
    ia=lookup[sc,mode,*aa];ib=lookup[sc,mode,*bb];image_checks.append({'scene':sc,'mode':mode,'pair':pairid,'equal_pixels_all9':bool(np.array_equal(images[ia],images[ib])),'different_pixels_by_frame':np.any(images[ia]!=images[ib],axis=-1).sum((1,2)).tolist(),'sampled_tool_state_maxabs':float(np.max(abs(states[ia]-states[ib])))})
 result={'scope':'Known-intervention pair geometry; no classifier/PCA; hidden force input; conditional 8-scene intervals','rows':rows,'summary':summary,'image_checks':image_checks,'t0_checks':t0,'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.data,a.features,Path(__file__)]}}
 (a.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('paired representation computed',len(rows))
if __name__=='__main__':main()
