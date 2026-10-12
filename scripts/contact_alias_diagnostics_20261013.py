"""G5h state-only contact/alias diagnostics; fixed phases, no fit/classifier."""
from pathlib import Path
import numpy as np,json,hashlib
R=Path('F:/2610/CVPR27');I=R/'runs/intervention_20261012/crossed_production_20261013';O=R/'runs/contact_alias_diagnostics_20261013';O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
z=np.load(I/'full_trajectories_with_repeats.npz');records=json.loads((I/'records.json').read_text());cfg=json.loads((I/'config.json').read_text());dt=cfg['clients'][0]['physics']['fixedTimeStep'];assert dt==.004;mass=np.array(cfg['mass_levels']);force=np.array(cfg['force_levels']);ratios=np.array([f/m for m in mass for f in force]);pairs=[(0,4),(3,7)];assert all(abs(ratios[a]-ratios[b])<1e-12 for a,b in pairs);lookup={(r['scene'],r['mass_kg'],r['force_n']):r for r in records if r['repeat']==0};S=np.zeros((40,9,121,13));contact=np.zeros((40,9),bool);normalmax=np.zeros((40,9));branchrows=[];validation=[]
for scene in range(40):
 for mi,m in enumerate(mass):
  for fi,f in enumerate(force):
   b=mi*3+fi;r=lookup[scene,float(m),float(f)];s=z[r['key']+'_fullstates'][:,0];assert s.shape==(121,13);S[scene,b]=s
   sample=[cp['normal_force'] for t in r['contact_details'] for client in t['clients'] for cp in client['contacts'] if cp['bodyB']==1];contact[scene,b]=any(v>1e-8 for v in sample);normalmax[scene,b]=max(sample,default=0)
   acc=np.diff(s[:,7:10],axis=0)/dt;integral=acc[:30].sum(0)*dt;err=float(np.max(np.abs(integral-(s[30,7:10]-s[0,7:10]))));assert err<1e-12;validation.append(err)
   dpos=np.diff(s[:,:3],axis=0);branchrows.append({'scene':scene,'split':'train' if scene<24 else 'tune' if scene<32 else 'dev','mass':float(m),'force':float(f),'force_over_mass':float(f/m),'sampled_positive_robot_contact':bool(contact[scene,b]),'sampled_robot_normal_force_max_N':float(normalmax[scene,b]),'acc_x_first5_mean_m_s2':float(acc[:5,0].mean()),'acc_x_on_mean_m_s2':float(acc[:30,0].mean()),'acc_on_vector_rms_m_s2':float(np.sqrt(np.mean(np.sum(acc[:30]**2,axis=1)))),'acc_x_on_minus_force_over_mass':float(acc[:30,0].mean()-f/m),'velocity30_m_s':s[30,7:10].tolist(),'velocity120_m_s':s[120,7:10].tolist(),'on_xy_path_m':float(np.linalg.norm(dpos[:30,:2],axis=1).sum()),'off_xy_path_m':float(np.linalg.norm(dpos[30:,:2],axis=1).sum()),'off_xy_endpoint_displacement_m':float(np.linalg.norm(s[120,:2]-s[30,:2]))})
 np.testing.assert_allclose(S[scene,:,0],np.repeat(S[scene,:1,0],9,axis=0),rtol=0,atol=1e-12)
position=S[:,:,:,:3];velocity=S[:,:,:,7:10];acc=np.diff(velocity,axis=2)/dt
# Distances retain physical units, with RMS across time of vector norm.
def distance(a,b):return float(np.sqrt(np.mean(np.sum((a-b)**2,axis=-1))))
phases={'on1_30':(position[:,:,1:31],velocity[:,:,1:31],acc[:,:,:30]),'off31_120':(position[:,:,31:],velocity[:,:,31:],acc[:,:,30:])};pairrows=[]
for scene in range(40):
 for pairid,(a,b) in enumerate(pairs):
  for phase,arrays in phases.items():
   for kind,A in zip(['position_m','velocity_m_s','acceleration_m_s2'],arrays):
    eq=distance(A[scene,a],A[scene,b]);controls=[]
    # Match scene, phase, metric and endpoint anchor. Non-ratio matches are not parameter-distance matched.
    for anchor in [a,b]:
     for other in range(9):
      if abs(ratios[anchor]-ratios[other])>1e-12:controls.append(distance(A[scene,anchor],A[scene,other]))
    ctrl=np.array(controls);pairrows.append({'scene':scene,'pair':pairid,'indices':[a,b],'ratio':float(ratios[a]),'phase':phase,'metric':kind,'equal_ratio_distance':eq,'different_ratio_median_distance':float(np.median(ctrl)),'different_ratio_min_distance':float(ctrl.min()),'fraction_different_ratio_distance_ge_equal':float(np.mean(ctrl>=eq)),'equal_over_different_median':float(eq/np.median(ctrl)) if np.median(ctrl)>0 else None,'either_sampled_robot_contact':bool(contact[scene,a] or contact[scene,b])})
summary=[];rng=np.random.default_rng(20261013);boot=rng.integers(40,size=(5000,40))
for pairid in [0,1]:
 for phase in phases:
  for kind in ['position_m','velocity_m_s','acceleration_m_s2']:
   rs=[r for r in pairrows if r['pair']==pairid and r['phase']==phase and r['metric']==kind];eq=np.array([r['equal_ratio_distance'] for r in rs]);diff=np.array([r['different_ratio_median_distance'] for r in rs]);summary.append({'pair':pairid,'phase':phase,'metric':kind,'equal_ratio_mean':float(eq.mean()),'equal_ratio_median':float(np.median(eq)),'different_ratio_median_per_scene_mean':float(diff.mean()),'paired_difference_diff_minus_equal_mean':float((diff-eq).mean()),'paired_scene_ci95':np.percentile((diff-eq)[boot].mean(1),[2.5,97.5]).tolist(),'equal_smaller_than_control_median_scene_count':int(np.sum(eq<diff)),'equal_distance_min':float(eq.min()),'equal_distance_max':float(eq.max())})
repeat=[]
for scene in range(40):
 rr=[r for r in records if r['scene']==scene and r['repeat']==1];assert len(rr)==1;r=rr[0];base=lookup[scene,.02,.02];repeat.append(float(np.max(np.abs(z[r['key']+'_fullstates']-z[base['key']+'_fullstates']))))
np.savez_compressed(O/'physical_trajectories.npz',state=S,acceleration=acc,ratio=ratios,sampled_robot_positive=contact,sampled_robot_normalmax=normalmax)
dump(O/'config.json',{'source_sha256':{str(p):sha(p) for p in [I/'full_trajectories_with_repeats.npz',I/'records.json',I/'config.json']},'code_sha256':sha(Path(__file__)),'dt':dt,'phases':['steps1..30 force applied','steps31..120 force off'],'equal_ratio_pairs':pairs,'ratios':ratios.tolist(),'control':'Same scene, phase, metric, and both endpoints as anchors; all different-ratio other branches. No parameter-distance match. All distances dependent within scene.','scope':'Existing40scene descriptive state diagnostics, not independent confirmation; no model training','contact_limit':'Only sampled steps1,5,10,20,30,60,90,120; no per-step force/directions/tangential force, cannot integrate full contact impulse.'});dump(O/'results.json',{'branches':branchrows,'pairs':pairrows,'summary':summary,'sampled_positive_robot_branches':int(contact.sum()),'sampled_positive_robot_scenes':int(contact.any(1).sum())});dump(O/'validation.json',{'status':'passed','velocity_acceleration_telescoping_max_error':max(validation),'baseline_repeat_fullstate_max_error':max(repeat),'all_initial_states_equal':True,'states_shape':list(S.shape)});print(json.dumps(summary,indent=2));print('robot',contact.sum(),contact.any(1).sum())
