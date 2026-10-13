"""Post-fit accounting only, no model refit or selection."""
import numpy as np
from family_anova_blocks_20261013 import R,O,components,construct,sha,dump
from pathlib import Path
with np.load(R/'pilot/dataset.npz') as f:z={k:f[k] for k in ['scene','codes','split']}
x=np.load(R/'features/features.npz')['grid'].astype(float)
b=np.load(O/'fitted.npz');U=b['U'];P=b['interaction_projector'];report={}
for domain,scenes in [('train',list(range(32))),('tune',list(range(32,40))),('dev',list(range(40,48)))]:
    energy=[];interaction={}
    for sc in scenes:
        X=x[z['scene']==sc];co=components(X.reshape(1,3,3,3,-1))
        centered=X-X.mean(0);den=np.sum(centered**2)
        row=[]
        for a in co:
            a=a.reshape(27,-1);e=float(np.sum(a*a));projected=a@U
            row.append([e,e/den,float(np.sum(projected**2)/e),float((e-np.sum(projected**2))/den)])
        energy.append(row);interaction[sc]=co[-1].reshape(27,-1)
    recs,_,_,_=construct(x,z,U,scenes);miss=[]
    for r in recs:
        I=interaction[r['scene']];delta=I[r['target']]-I[r['a']]
        full_ratio=np.sum(delta**2,axis=1)/r['den']
        held_shared=np.sum((r['t']@P)**2,axis=1)/r['den']
        pca_omitted=1-np.sum(r['t']**2,axis=1)/r['den']
        miss.append([full_ratio.mean(),held_shared.mean(),pca_omitted.mean()])
    report[domain]={'component_order':['position_main','color_main','background_main','interaction_2plus3'],
      'energy_columns':['full_energy','fraction_of_centered_grid_energy','pca_retained_fraction_of_component','pca_omitted_fraction_of_total_grid'],
      'energy_scene_mean':np.mean(energy,axis=0).tolist(),
      'miss_columns':['true_interaction_delta_energy_over_total_target_delta','unchanged_shared_block_target_energy_over_total','pca_omitted_target_energy_over_total'],
      'miss_by_factor':np.asarray(miss).reshape(-1,3,3).mean(0).tolist(),
      'scope':'truth-based scoring only; no test-grid quantity enters fitted map or editor; interaction and main delta cross terms need not vanish for an individual edit'}
dump(O/'anova_energy_and_omission.json',report)
dump(O/'manifest.json',{str(p):sha(p) for p in list(O.iterdir())+[Path(__file__),Path(__file__).with_name('family_anova_blocks_20261013.py'),Path(__file__).with_name('fit_target_supervised_reference_20261012.py')] if p.is_file() and p.name!='manifest.json'})
print(report)
allocation={}
ps=np.concatenate([b['joint0'],(np.eye(64)-b['joint0'].sum(0))[None]])
for domain,scenes in [('train',list(range(32))),('tune',list(range(32,40))),('dev',list(range(40,48)))]:
    co=components(np.stack([(x[z['scene']==sc]@U).reshape(3,3,3,64) for sc in scenes]))
    allocation[domain]=[[float(np.mean([np.sum((a@p)**2)/np.sum(a*a) for a in arr.reshape(len(scenes),27,64)])) for p in ps] for arr in co]
dump(O/'joint0_component_allocation.json',allocation)
a=np.load(O/'dev_anova.npz')['scene_metrics'];j=np.load(O/'dev_joint0.npz')['scene_metrics'];d=((a-j)[:4]+(a-j)[4:])/2
dump(O/'dev_pair_differences.json',{'unit':'four disjoint scene pairs, both orientations averaged; descriptive range not CI','factor_metric_mean':d.mean(0).tolist(),'factor_metric_min':d.min(0).tolist(),'factor_metric_max':d.max(0).tolist()})
dump(O/'manifest.json',{str(p):sha(p) for p in list(O.iterdir())+[Path(__file__),Path(__file__).with_name('family_anova_blocks_20261013.py'),Path(__file__).with_name('fit_target_supervised_reference_20261012.py')] if p.is_file() and p.name!='manifest.json'})
