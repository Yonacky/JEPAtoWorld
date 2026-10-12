"""G5a production:40 scene groups,7 unique branches,9 frames; no encoder."""
import argparse,pathlib,json,hashlib,traceback,itertools
import numpy as np
import pybullet as pb
from causal_world.envs import CausalWorld
from causal_world.task_generators.task import generate_task
from causal_world.configs.world_constants import WorldConstants

def dump(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg={'seed':20261015,'scenes':40,'future_full_split_plan':[24,8,8],'mass_levels':[.01,.02,.04],'friction_levels':[.05,.175,.6],'force_levels':[.01,.02,.03],'baseline':{'mass':.02,'friction':.175,'force':.02},'steps':120,'force_steps':30,'record_steps':[0,1,5,10,20,30,60,90,120],'force':'world+x at instantaneous object center of mass; same sequence for mass/friction branches','scope':'raw Bullet CPU dynamics inside existing CausalWorld pushing fixture; not env.step policy replay; no encoder','source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()};dump(a.output/'config.json',cfg)
 e=None;records=[];arrays={};checks=[];learning=[]
 try:
  e=CausalWorld(task=generate_task(task_generator_id='pushing'),observation_mode='pixel',enable_visualization=False,seed=cfg['seed']);e.reset();obj=e._stage.get_rigid_objects()['tool_block'];clients=list(obj._pybullet_client_ids);bodies=list(obj._block_ids);assert len(clients)==len(set(clients))
  cfg['clients']=[{'client':int(c),'body':int(b),'physics':pb.getPhysicsEngineParameters(physicsClientId=c)} for c,b in zip(clients,bodies)];cfg['env_clients']={k:getattr(e,k,None) for k in ['_pybullet_client_w_o_goal_id','_pybullet_client_w_goal_id','_pybullet_client_full_id']};dump(a.output/'config.json',cfg)
  def state():
   result=[]
   for c,b in zip(clients,bodies):
    pos,q=pb.getBasePositionAndOrientation(b,physicsClientId=c);vel,ang=pb.getBaseVelocity(b,physicsClientId=c);result.append(np.r_[pos,q,vel,ang])
   return np.array(result)
  def render():
   before=state();im=e._robot.get_current_camera_observations()[0];np.testing.assert_allclose(state(),before,rtol=0,atol=1e-12);return np.rint(im*255).astype(np.uint8)
  rng=np.random.default_rng(cfg['seed'])
  for scene in range(40):
   # Re-establish identical geometry/control within each scene, then save all client worlds.
   joints=np.tile([0.,np.pi/4,-np.pi/4],3);e._robot.apply_interventions({'joint_positions':joints,'joint_velocities':np.zeros(9)})
   e._stage.apply_interventions({'floor_friction':.2,'stage_friction':.3,'tool_block':{'mass':.02}});e._stage.apply_interventions({'tool_block':{'friction':.175}})
   xy=[float(rng.uniform(-.035,-.015)),float(rng.uniform(-.025,.025))];yaw=float(rng.uniform(-.2,.2));quat=pb.getQuaternionFromEuler([0,0,yaw])
   for c,b in zip(clients,bodies):
    pb.resetBasePositionAndOrientation(b,[xy[0],xy[1],.0325+WorldConstants.FLOOR_HEIGHT],quat,physicsClientId=c);pb.resetBaseVelocity(b,[0,0,0],[0,0,0],physicsClientId=c)
   for _ in range(120):
    for c in clients:pb.stepSimulation(physicsClientId=c)
   # The same resting baseline is restored for every branch, including solver state.
   for c,b in zip(clients,bodies):pb.resetBaseVelocity(b,[0,0,0],[0,0,0],physicsClientId=c)
   initial=state();baseline_dyn=[pb.getDynamicsInfo(b,-1,physicsClientId=c) for c,b in zip(clients,bodies)];snapshots=[pb.saveState(physicsClientId=c) for c in clients];reference=render();assert np.array_equal(render(),reference)
   scene_entries={}
   def branch(factor,level,repeat):
    for c,s in zip(clients,snapshots):pb.restoreState(stateId=s,physicsClientId=c)
    mass=level if factor=='mass' else .02;friction=level if factor=='friction' else .175;force=level if factor=='force' else .02
    # Stage setters separately: legacy mass/friction elif; reset internal caches on every branch.
    e._stage.apply_interventions({'tool_block':{'mass':mass}});e._stage.apply_interventions({'tool_block':{'friction':friction}})
    dynamics=[]
    for c,b,bd in zip(clients,bodies,baseline_dyn):
     # Constant shape/density change: inertia scales with mass; never leave stale inertia.
     pb.changeDynamics(b,-1,localInertiaDiagonal=(np.array(bd[2])*mass/.02).tolist(),physicsClientId=c)
     d=pb.getDynamicsInfo(b,-1,physicsClientId=c);assert abs(d[0]-mass)<1e-12 and abs(d[1]-friction)<1e-12;dynamics.append({'client':c,'mass':d[0],'lateral_friction':d[1],'local_inertia':d[2]})
    np.testing.assert_allclose(state(),initial,rtol=0,atol=1e-12);im0=render();assert np.array_equal(im0,reference),'initial frame changed by invisible factor'
    images=[im0];states=[state()];fullstates=[state()];contact=[]
    for step in range(1,cfg['steps']+1):
     for c,b in zip(clients,bodies):
      if step<=cfg['force_steps']:
       pos=pb.getBasePositionAndOrientation(b,physicsClientId=c)[0];pb.applyExternalForce(b,-1,[force,0,0],pos,pb.WORLD_FRAME,physicsClientId=c)
      pb.stepSimulation(physicsClientId=c)
     fullstates.append(state())
     if step in cfg['record_steps']:
      states.append(state());images.append(render());contact.append([len(pb.getContactPoints(bodyA=b,physicsClientId=c)) for c,b in zip(clients,bodies)])
    key=f's{scene}_{factor}_{level}_{repeat}';arrays[key+'_images']=np.array(images);arrays[key+'_states']=np.array(states);arrays[key+'_fullstates']=np.array(fullstates);records.append({'key':key,'scene':scene,'factor':factor,'level':level,'repeat':repeat,'actual_dynamics':dynamics,'force_n':force,'sample_steps':cfg['record_steps'],'contact_counts':contact,'initial_image_exact':True,'initial_state_exact':True});return key
   baseline=branch('baseline',.02,0);learning.append((scene,'baseline',1,baseline,[.02,.175,.02]))
   second=branch('baseline',.02,1);repeaterr=float(np.max(abs(arrays[baseline+'_fullstates']-arrays[second+'_fullstates'])));repeatpix=int(np.count_nonzero(arrays[baseline+'_images']!=arrays[second+'_images']));assert repeaterr<1e-10 and repeatpix==0,('restore/repeat mismatch',scene,repeaterr,repeatpix)
   for factor,levels in [('mass',cfg['mass_levels']),('friction',cfg['friction_levels']),('force',cfg['force_levels'])]:
    low=branch(factor,float(levels[0]),0);high=branch(factor,float(levels[2]),0);keys=[low,baseline,high]
    for lev,key in [(0,low),(2,high)]:
     physical=[.02,.175,.02];physical[['mass','friction','force'].index(factor)]=levels[lev];learning.append((scene,factor,lev,key,physical))
    diff=arrays[low+'_fullstates']-arrays[high+'_fullstates'];maxdist=float(np.linalg.norm(diff[:,:,0:3],axis=-1).max());pixels=int(np.any(arrays[low+'_images'][-1]!=arrays[high+'_images'][-1],axis=-1).sum());checks.append({'scene':scene,'factor':factor,'repeat_state_maxabs':repeaterr,'repeat_pixels':repeatpix,'extreme_future_position_maxdiff':maxdist,'extreme_final_changed_pixels':pixels,'effect_visible':bool(maxdist>1e-4 and pixels>=5)})
   for c,s in zip(clients,snapshots):pb.removeState(s,physicsClientId=c)
   print('scene',scene,'checks',checks[-3:],flush=True)
  np.savez_compressed(a.output/'full_trajectories_with_repeats.npz',**arrays);dump(a.output/'records.json',records)
  dataset={k:[] for k in ['images','labels','split','episode','frame','episode_length','scene','factor','level','actual_states','time_seconds']}
  timesteps=[float(pb.getPhysicsEngineParameters(physicsClientId=c)['fixedTimeStep']) for c in clients];assert np.max(timesteps)-np.min(timesteps)<1e-12
  for scene,factor,level,key,physical in learning:
   for j,step in enumerate(cfg['record_steps']):
    row={'images':arrays[key+'_images'][j],'labels':physical,'split':0 if scene<24 else 1 if scene<32 else 2,'episode':key,'frame':step,'episode_length':121,'scene':scene,'factor':factor,'level':level,'actual_states':arrays[key+'_states'][j],'time_seconds':step*timesteps[0]}
    for k,v in row.items():dataset[k].append(v)
  dataset={k:np.asarray(v) for k,v in dataset.items()};dataset['target_names']=np.array(['mass_kg','lateral_friction','force_N']);dataset['state_names']=np.array(['x','y','z','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz']);assert len(dataset['images'])==2520
  # Do not release failed-effect branches as training data: preserve raw failures only.
  passed=all(c['effect_visible'] for c in checks);filename='dataset.npz' if passed else 'FAILED_EFFECT_dataset_DO_NOT_TRAIN.npz';np.savez_compressed(a.output/filename,**dataset)
  dump(a.output/'validation.json',{'status':'PASS' if passed else 'EFFECT_INSUFFICIENT','checks':checks,'dataset_filename':filename,'dataset_sha256':hashlib.sha256((a.output/filename).read_bytes()).hexdigest(),'frames':2520,'unique_learning_branches':280,'repeat_branches_excluded':40,'scene_split':[24,8,8],'physics_timestep':timesteps,'interpretation':'scene-grouped physical branch data; no model trained; t0 identical across7 branches; baseline stored only once'})
 except Exception as ex:
  if arrays:np.savez_compressed(a.output/'partial_trajectories.npz',**arrays)
  dump(a.output/'FAILED.json',{'error':str(ex),'traceback':traceback.format_exc(),'records':records,'checks':checks});raise
 finally:
  if e is not None:e.close()
if __name__=='__main__':main()
