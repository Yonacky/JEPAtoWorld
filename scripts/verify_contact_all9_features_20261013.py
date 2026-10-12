"""G5i consistency audit across all nine frozen sample times, not a new metric."""
from pathlib import Path
import argparse,numpy as np,json,hashlib
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();z=np.load(a.data);f=np.load(a.features);arrays={s:f[s] for s in ['mean','grid']};rows=[]
for mode in ['on','off']:
 for sc in range(40):
  for pi,(aa,bb) in enumerate([((0,0),(1,1)),((1,0),(2,1))]):
   ids=[]
   for cell in [aa,bb]:
    ix=np.flatnonzero((z['scene']==sc)&(z['contact_mode']==mode)&(z['mass_level']==cell[0])&(z['force_level']==cell[1]));ix=ix[np.argsort(z['frame'][ix])];np.testing.assert_array_equal(z['frame'][ix],[0,1,5,10,20,30,60,90,120]);ids.append(ix)
   for source,A in arrays.items():
    x,y=A[ids[0]],A[ids[1]];rows.append({'scene':sc,'mode':mode,'pair':pi,'source':source,'all9_exact':bool(np.array_equal(x,y)),'exact_by_time':np.all(x==y,axis=1).tolist(),'maxabs_by_time':np.max(abs(x.astype(float)-y.astype(float)),axis=1).tolist()})
summary={mode:{source:sum(r['all9_exact'] for r in rows if r['mode']==mode and r['source']==source) for source in ['mean','grid']} for mode in ['on','off']};a.output.mkdir(exist_ok=True,parents=True);(a.output/'all9_feature_equality.json').write_text(json.dumps({'status':'audited','pairs_per_mode_source':80,'summary_exact_pairs':summary,'rows':rows,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8');print(summary)
