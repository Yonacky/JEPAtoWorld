"""G5h paired ON/OFF physical mechanism diagnostic, no learned model."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();I=args.input;O=args.output;O.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def rms(x):return float(np.linalg.norm(x.ravel())/np.sqrt(x.shape[0]))
records=json.loads((I/'records.json').read_text());meta=json.loads((I/'scene_metadata.json').read_text());cfg=json.loads((I/'config.json').read_text());raw=np.load(I/'full_trajectories_with_repeats.npz');N=cfg['scenes'];M=cfg['mass_levels'];F=cfg['force_levels'];ratio=np.array([f/m for m in M for f in F]);pairs=[(0,4),(3,7)];S=np.zeros((N,2,9,121,13));lookup={(r['scene'],r['contact_mode'],r['mass_kg'],r['force_n']):r for r in records if r['repeat']==0};branch=[];paired=[];equal=[];repeats=[];Jall=[];robotstates=[];initial_errors=[]
for scene in range(N):
 mapping=meta[scene]['runtime_body_mapping'];assert len(mapping)==1;mp=mapping[0];dt=mp['physics']['fixedTimeStep'];assert dt==.004;robot=mp['robot'];floor=mp['floor'];tool=mp['tool'];robscene=[]
 for modeid,mode in enumerate(['on','off']):
  robmode=[]
  for mi,mass in enumerate(M):
   for fi,force in enumerate(F):
    b=mi*3+fi;r=lookup[scene,mode,mass,force];key=r['key'];st=raw[key+'_fullstates'][:,0];rs=raw[key+'_robot_states'][:,0];assert st.shape==(121,13) and rs.shape[0]==121;S[scene,modeid,b]=st;robmode.append(rs);assert r['explicit_filter_enable']==int(mode=='on');assert r['filter_robot_links']==meta[scene]['filter_robot_links'];assert [t['step'] for t in r['contacts_every_step']]==list(range(121))
    forces=np.zeros((120,3,3));counts=np.zeros((120,3),int);min_distance=0.
    for t in r['contacts_every_step'][1:]:
     for cl in t['clients']:
      assert cl['tool']==tool and cl['client']==mp['client']
      for c in cl['contacts']:
       assert c['bodyA']==tool;g=0 if c['bodyB']==robot else 1 if c['bodyB']==floor else 2;fv=np.array(c['normal_on_B'])*c['normal_force']+np.array(c['lateral_direction1'])*c['lateral_force1']+np.array(c['lateral_direction2'])*c['lateral_force2'];forces[t['step']-1,g]+=fv;counts[t['step']-1,g]+=int(c['normal_force']>1e-8);min_distance=min(min_distance,c['distance'])
    if mode=='off':assert counts[:,0].sum()==0 and np.max(np.abs(forces[:,0]))<=1e-8
    assert counts[:,1].sum()>0;acc=np.diff(st[:,7:10],axis=0)/dt;step_error=float(np.max(np.abs(acc.sum(0)*dt-(st[-1,7:10]-st[0,7:10]))));assert step_error<1e-12
    row={'scene':scene,'split':'dev' if N==40 and scene>=32 else 'tune' if N==40 and scene>=24 else 'train_or_audit','mode':mode,'mass':mass,'force':force,'ratio':force/mass,'robot_positive_step_count':int(np.count_nonzero(counts[:,0])),'floor_positive_step_count':int(np.count_nonzero(counts[:,1])),'minimum_contact_distance':min_distance,'robot_position_drift_max':float(np.max(np.abs(rs[:,:,0]-rs[0,:,0]))),'robot_velocity_drift_max':float(np.max(np.abs(rs[:,:,1]-rs[0,:,1]))),'all_state_finite':bool(np.isfinite(st).all()),'off_xy_path_m':float(np.linalg.norm(np.diff(st[30:,:2],axis=0),axis=1).sum()),'early5_accx_mean':float(acc[:5,0].mean()),'on_accx_mean':float(acc[:30,0].mean())}
    gravity=np.array([mp['physics']['gravityAccelerationX'],mp['physics']['gravityAccelerationY'],mp['physics']['gravityAccelerationZ']])
    for phase,sl,t0,t1 in [('on',slice(0,30),0,30),('off',slice(30,120),30,120)]:
     impulse=forces[sl].sum(0)*dt;external=np.array([force*(30*dt) if phase=='on' else 0.,0.,0.]);weight=mass*gravity*((t1-t0)*dt);momentum=mass*(st[t1,7:10]-st[t0,7:10]);residual=momentum-impulse.sum(0)-external-weight;row[phase+'_contact_impulse_robot_floor_other_Ns']=impulse.tolist();row[phase+'_momentum_change_kgm_s']=momentum.tolist();row[phase+'_recorded_force_bookkeeping_residual_Ns']=residual.tolist()
    branch.append(row);Jall.append(forces)
  robscene.append(np.array(robmode))
  repeat=next(r for r in records if r['scene']==scene and r['contact_mode']==mode and r['repeat']==1);base=lookup[scene,mode,.02,.02];repeats.append(float(np.max(np.abs(raw[repeat['key']+'_fullstates']-raw[base['key']+'_fullstates']))));assert repeats[-1]<1e-10
 robotstates.append(np.array(robscene));initial_errors.append(float(np.max(np.abs(S[scene,:, :,0]-S[scene,0,0,0]))));assert initial_errors[-1]<1e-12
 for b in range(9):
  aa=S[scene,0,b];bb=S[scene,1,b];ra=robscene[0][b];rb=robscene[1][b];assert np.max(np.abs(ra[0]-rb[0]))<1e-12
  for phase,sl in [('on',slice(1,31)),('off',slice(31,121))]:paired.append({'scene':scene,'branch':b,'phase':phase,'position_ON_OFF_rms_m':rms(aa[sl,:3]-bb[sl,:3]),'velocity_ON_OFF_rms_m_s':rms(aa[sl,7:10]-bb[sl,7:10]),'robot_joint_position_ON_OFF_rms':rms(ra[sl,:,0]-rb[sl,:,0]),'robot_joint_velocity_ON_OFF_rms':rms(ra[sl,:,1]-rb[sl,:,1])})
 for pairid,(a,b) in enumerate(pairs):
  for phase,sl in [('on',slice(1,31)),('off',slice(31,121))]:
   for kind,cols in [('position_m',slice(0,3)),('velocity_m_s',slice(7,10))]:
    vals=[]
    for modeid,mode in enumerate(['on','off']):
     va=S[scene,modeid,a,sl,cols];vb=S[scene,modeid,b,sl,cols];eq=rms(va-vb);ctrl=[rms(S[scene,modeid,anchor,sl,cols]-S[scene,modeid,other,sl,cols]) for anchor in [a,b] for other in range(9) if abs(ratio[anchor]-ratio[other])>1e-12];vals.append(eq);thresholds=[1e-6,1e-5,1e-4] if kind=='position_m' else [1e-5,1e-4,1e-3];equal.append({'scene':scene,'pair':pairid,'phase':phase,'metric':kind,'mode':mode,'equal_ratio_rms':eq,'equal_ratio_max_vector_norm':float(np.linalg.norm(va-vb,axis=1).max()),'different_ratio_median':float(np.median(ctrl)),'near_equivalent_rms_thresholds':{str(t):bool(eq<t) for t in thresholds}})
    equal.append({'scene':scene,'pair':pairid,'phase':phase,'metric':kind,'mode':'off_minus_on','equal_ratio_rms':vals[1]-vals[0]})
np.savez_compressed(O/'states_for_recompute.npz',state=S,robot_state=np.array(robotstates),contact_force_robot_floor_other=np.array(Jall),ratio=ratio)
summary=[];rng=np.random.default_rng(20261018)
for split,scenes in [('all',np.arange(N)),('dev',np.arange(32,N) if N==40 else np.array([],int))]:
 if len(scenes)==0:continue
 bi=rng.integers(len(scenes),size=(5000,len(scenes)))
 for pairid in [0,1]:
  for phase in ['on','off']:
   for metric in ['position_m','velocity_m_s']:
    row={'scene_set':split,'n_scenes':len(scenes),'pair':pairid,'phase':phase,'metric':metric}
    for mode in ['on','off','off_minus_on']:
     vv=np.array([next(r['equal_ratio_rms'] for r in equal if r['scene']==s and r['pair']==pairid and r['phase']==phase and r['metric']==metric and r['mode']==mode) for s in scenes]);row[mode+'_mean']=float(vv.mean());row[mode+'_scene_ci95']=np.percentile(vv[bi].mean(1),[2.5,97.5]).tolist()
    summary.append(row)
dump(O/'config.json',{'input':str(I),'source_sha256':{str(p):sha(p) for p in [I/'records.json',I/'scene_metadata.json',I/'full_trajectories_with_repeats.npz',I/'config.json']},'code_sha256':sha(Path(__file__)),'scope':'Physical collision-removal paired diagnostic, includes release transient from ON-settled snapshot; no classifier','impulse_convention':'sum normal_on_B*normal_force+lateral_direction1*force1+lateral_direction2*force2 as recorded on tool A, steps1..120*dt; report momentum residual, not guaranteed complete force balance','split':'all descriptive; dev32..39 separately for40-scene production','equal_ratio_pairs':pairs});dump(O/'results.json',{'branches':branch,'on_off_paired':paired,'equal_ratio':equal,'summary':summary});dump(O/'validation.json',{'status':'passed','initial_tool_error':max(initial_errors),'repeat_state_error':max(repeats),'off_robot_positive_forces_absent':True,'floor_present_every_branch':True});print(json.dumps(summary,indent=2))

# Independent reductions use coordinate sums rather than the production flat-norm routine.
q=np.load(O/'states_for_recompute.npz');zs=q['state'];zr=q['robot_state'];err=[]
for r in equal:
 if r['mode']=='off_minus_on':continue
 a,b=pairs[r['pair']];sl=slice(1,31) if r['phase']=='on' else slice(31,121);cols=slice(0,3) if r['metric']=='position_m' else slice(7,10);d=zs[r['scene'],int(r['mode']=='off'),a,sl,cols]-zs[r['scene'],int(r['mode']=='off'),b,sl,cols];v=float(np.sqrt(np.sum(d*d,axis=1).mean()));err.append(abs(v-r['equal_ratio_rms']))
full_equivalence=[]
for sc in range(N):
 for md,mode in enumerate(['on','off']):
  for pid,(a,b) in enumerate(pairs):
   full_equivalence.append({'scene':sc,'mode':mode,'pair':pid,'full_state_max_abs':float(np.max(np.abs(zs[sc,md,a]-zs[sc,md,b]))),'robot_state_max_abs':float(np.max(np.abs(zr[sc,md,a]-zr[sc,md,b])))})
force_reduction_errors=[]
forces=q['contact_force_robot_floor_other']
for i,r in enumerate(branch):
 for phase,lo,hi in [('on',0,30),('off',30,120)]:
  independently=np.zeros((3,3))
  for step in range(lo,hi):independently+=forces[i,step]*.004
  force_reduction_errors.append(float(np.max(np.abs(independently-np.asarray(r[phase+'_contact_impulse_robot_floor_other_Ns'])))))
assert max(err)<1e-12 and max(force_reduction_errors)<1e-12
dump(O/'independent_recompute.json',{'status':'passed','distance_max_error':max(err),'contact_impulse_reduction_max_error':max(force_reduction_errors),'scope':'Independent reductions of saved states/contact vectors; source contact vector convention separately checked by momentum residual. Not independent simulator replay.','full_state_equivalence':full_equivalence})
