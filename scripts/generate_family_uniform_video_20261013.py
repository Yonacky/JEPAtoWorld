"""G7 uniform 16-frame replay; physics unchanged from G5f. CPU simulation."""
import argparse,pathlib,json,hashlib,traceback,itertools
import numpy as np
import pybullet as pb
from causal_world.envs import CausalWorld
from causal_world.task_generators.task import generate_task
from causal_world.configs.world_constants import WorldConstants
STEPS=list(range(0,121,8))
def dump(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--scenes',type=int,choices=[2,40],default=2);ap.add_argument('--seed',type=int,default=20261018);a=ap.parse_args();assert (a.scenes,a.seed) in [(2,20261017),(40,20261018)],'frozen audit2 seed17 or production40 seed18';a.output.mkdir(parents=True,exist_ok=False)
 cfg={'seed':a.seed,'scenes':a.scenes,'mass_levels':[.01,.02,.04],'force_levels':[.01,.02,.03],'friction':.175,'modes':['on','off'],'record_steps':STEPS,'steps':120,'force_steps':30,'scene_split':[24,8,8] if a.scenes==40 else [2,0,0],'fresh_environment_per_scene':True,'settling':'120 steps with robot-tool collisions ON, save one snapshot shared by both modes','collision_intervention':'all robot links including base against tool base; robot appearance and control settings unchanged','scope':'collision-removal counterfactual, not a physically absent robot; possible initial interpenetration logged','source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()};dump(a.output/'config.json',cfg)
 rng=np.random.default_rng(cfg['seed']);arrays={};records=[];scene_meta=[];learning=[];checks=[];env=None
 try:
  for scene in range(a.scenes):
   env=CausalWorld(task=generate_task(task_generator_id='pushing'),observation_mode='pixel',enable_visualization=False,seed=cfg['seed']+scene);env.reset();obj=env._stage.get_rigid_objects()['tool_block'];clients=list(obj._pybullet_client_ids);bodies=list(obj._block_ids);robot=WorldConstants.ROBOT_ID;floor=WorldConstants.FLOOR_ID;assert len(clients)==len(set(clients))
   mappings=[];links=[];templates=[]
   for c,b in zip(clients,bodies):
    bodyinfo=[]
    for i in range(pb.getNumBodies(physicsClientId=c)):
     bid=pb.getBodyUniqueId(i,physicsClientId=c);info=pb.getBodyInfo(bid,physicsClientId=c);bodyinfo.append({'id':bid,'base_name':info[0].decode('utf-8'),'body_name':info[1].decode('utf-8'),'num_joints':pb.getNumJoints(bid,physicsClientId=c)})
    assert pb.getNumJoints(robot,physicsClientId=c)>0 and pb.getNumJoints(b,physicsClientId=c)==0
    mappings.append({'client':c,'tool':b,'robot':robot,'floor':floor,'body_info':bodyinfo,'physics':pb.getPhysicsEngineParameters(physicsClientId=c)});links.append(list(range(-1,pb.getNumJoints(robot,physicsClientId=c))));templates.append(pb.getDynamicsInfo(b,-1,physicsClientId=c))
   inertia_per_mass=[np.array(d[2])/d[0] for d in templates]
   def filters(mode):
    for c,b,ll in zip(clients,bodies,links):
     for link in ll:pb.setCollisionFilterPair(robot,b,link,-1,int(mode=='on'),physicsClientId=c)
   def state():
    out=[]
    for c,b in zip(clients,bodies):
     pos,q=pb.getBasePositionAndOrientation(b,physicsClientId=c);v,w=pb.getBaseVelocity(b,physicsClientId=c);out.append(np.r_[pos,q,v,w])
    return np.array(out)
   def robotstate():
    return np.array([[list(s[:2]) for s in pb.getJointStates(robot,list(range(pb.getNumJoints(robot,physicsClientId=c))),physicsClientId=c)] for c in clients])
   def worldstate():
    out=[]
    for c in clients:
     for i in range(pb.getNumBodies(physicsClientId=c)):
      b=pb.getBodyUniqueId(i,physicsClientId=c);pos,q=pb.getBasePositionAndOrientation(b,physicsClientId=c);v,w=pb.getBaseVelocity(b,physicsClientId=c);out.append(np.r_[c,b,pos,q,v,w])
    return np.array(out)
   def render():
    before=state();im=env._robot.get_current_camera_observations()[0];np.testing.assert_allclose(state(),before,atol=1e-12,rtol=0);return np.rint(im*255).astype(np.uint8)
   def contacts(step):
    return {'step':step,'clients':[{'client':c,'tool':b,'contacts':[{'bodyA':int(p[1]),'bodyB':int(p[2]),'linkA':int(p[3]),'linkB':int(p[4]),'position_on_A':list(p[5]),'position_on_B':list(p[6]),'normal_on_B':list(p[7]),'lateral_direction1':list(p[11]),'lateral_direction2':list(p[13]),'distance':float(p[8]),'normal_force':float(p[9]),'lateral_force1':float(p[10]),'lateral_force2':float(p[12])} for p in pb.getContactPoints(bodyA=b,physicsClientId=c)]} for c,b in zip(clients,bodies)]}
   filters('on');env._robot.apply_interventions({'joint_positions':np.tile([0.,np.pi/4,-np.pi/4],3),'joint_velocities':np.zeros(9)})
   env._stage.apply_interventions({'floor_friction':.2,'stage_friction':.3,'tool_block':{'mass':.02}});env._stage.apply_interventions({'tool_block':{'friction':.175}})
   xy=[float(rng.uniform(-.035,-.015)),float(rng.uniform(-.025,.025))];yaw=float(rng.uniform(-.2,.2));quat=pb.getQuaternionFromEuler([0,0,yaw])
   for c,b,ipm in zip(clients,bodies,inertia_per_mass):
    pb.changeDynamics(b,-1,localInertiaDiagonal=(ipm*.02).tolist(),physicsClientId=c);pb.resetBasePositionAndOrientation(b,[*xy,.0325+WorldConstants.FLOOR_HEIGHT],quat,physicsClientId=c);pb.resetBaseVelocity(b,[0,0,0],[0,0,0],physicsClientId=c)
   for _ in range(120):
    for c in clients:pb.stepSimulation(physicsClientId=c)
   for c,b in zip(clients,bodies):pb.resetBaseVelocity(b,[0,0,0],[0,0,0],physicsClientId=c)
   initial=state();initialrobot=robotstate();initialworld=worldstate();reference=render();saved=[pb.saveState(physicsClientId=c) for c in clients]
   scene_meta.append({'scene':scene,'env_seed':cfg['seed']+scene,'xy_requested':xy,'yaw_requested':yaw,'runtime_body_mapping':mappings,'inertia_source_dynamics_info':[list(d) for d in templates],'inertia_per_mass':[v.tolist() for v in inertia_per_mass],'initial_world_body_states':initialworld.tolist(),'initial_robot_joint_states':initialrobot.tolist(),'initial_contacts_on':contacts(0),'filter_robot_links':links})
   def branch(mode,mass,force,repeat):
    for c,s in zip(clients,saved):pb.restoreState(stateId=s,physicsClientId=c)
    env._stage.apply_interventions({'tool_block':{'mass':mass}});env._stage.apply_interventions({'tool_block':{'friction':.175}});filters(mode)
    dynamics=[]
    for c,b,ipm in zip(clients,bodies,inertia_per_mass):
     pb.changeDynamics(b,-1,localInertiaDiagonal=(ipm*mass).tolist(),physicsClientId=c);d=pb.getDynamicsInfo(b,-1,physicsClientId=c);assert abs(d[0]-mass)<1e-12 and abs(d[1]-.175)<1e-12;np.testing.assert_allclose(d[2],ipm*mass,atol=1e-12,rtol=0);dynamics.append({'client':c,'mass':d[0],'friction':d[1],'inertia':d[2]});pb.performCollisionDetection(physicsClientId=c)
    np.testing.assert_allclose(state(),initial,atol=1e-12,rtol=0);np.testing.assert_allclose(robotstate(),initialrobot,atol=1e-12,rtol=0);np.testing.assert_allclose(worldstate(),initialworld,atol=1e-12,rtol=0);im0=render();np.testing.assert_array_equal(im0,reference)
    ims=[im0];ss=[state()];rs=[robotstate()];cc=[contacts(0)]
    for step in range(1,121):
     for c,b in zip(clients,bodies):
      if step<=30:pb.applyExternalForce(b,-1,[force,0,0],pb.getBasePositionAndOrientation(b,physicsClientId=c)[0],pb.WORLD_FRAME,physicsClientId=c)
      pb.stepSimulation(physicsClientId=c)
     ss.append(state());rs.append(robotstate());cc.append(contacts(step))
     if step in STEPS:ims.append(render())
    maxrobot=max([abs(p['normal_force']) for t in cc for cl in t['clients'] for p in cl['contacts'] if p['bodyB']==robot]+[0.]);floorpositive=sum(p['normal_force']>1e-8 for t in cc[1:] for cl in t['clients'] for p in cl['contacts'] if p['bodyB']==floor)
    if mode=='off':assert maxrobot<=1e-8,('OFF robot contact force',maxrobot)
    assert floorpositive>0,('floor contact absent',mode,mass,force)
    key=f's{scene}_{mode}_m{mass}_f{force}_r{repeat}';arrays[key+'_images']=np.array(ims);arrays[key+'_fullstates']=np.array(ss);arrays[key+'_robot_states']=np.array(rs);records.append({'key':key,'scene':scene,'contact_mode':mode,'mass_kg':mass,'force_n':force,'repeat':repeat,'actual_dynamics':dynamics,'explicit_filter_enable':int(mode=='on'),'filter_robot_links':links,'contacts_every_step':cc,'off_robot_force_max':maxrobot,'floor_positive_entries':int(floorpositive)});return key
   grid={}
   for mode in ['on','off']:
    for mi,mass in enumerate(cfg['mass_levels']):
     for fi,force in enumerate(cfg['force_levels']):
      key=branch(mode,mass,force,0);grid[(mode,mi,fi)]=key;learning.append((scene,mode,mi,fi,key,[mass,.175,force]))
    repeat=branch(mode,.02,.02,1);baseline=grid[(mode,1,1)];err=float(np.max(abs(arrays[baseline+'_fullstates']-arrays[repeat+'_fullstates'])));rerr=float(np.max(abs(arrays[baseline+'_robot_states']-arrays[repeat+'_robot_states'])));assert max(err,rerr)<1e-10;np.testing.assert_array_equal(arrays[baseline+'_images'],arrays[repeat+'_images']);checks.append({'scene':scene,'mode':mode,'repeat_tool_maxabs':err,'repeat_robot_maxabs':rerr})
   for mi,fi in itertools.product(range(3),repeat=2):
    on,off=[grid[(mode,mi,fi)] for mode in ['on','off']];np.testing.assert_array_equal(arrays[on+'_images'][0],arrays[off+'_images'][0]);np.testing.assert_allclose(arrays[on+'_fullstates'][0],arrays[off+'_fullstates'][0],atol=1e-12,rtol=0)
   for c,s in zip(clients,saved):pb.removeState(s,physicsClientId=c)
   env.close();env=None;print('scene',scene,'paired9 PASS',flush=True)
  dataset={k:[] for k in ['images','labels','split','episode','frame','episode_length','scene','contact_mode','mass_level','force_level','actual_states','time_seconds']}
  dt=scene_meta[0]['runtime_body_mapping'][0]['physics']['fixedTimeStep'];assert abs(dt-.004)<1e-12
  for scene,mode,mi,fi,key,label in learning:
   for j,step in enumerate(STEPS):
    row={'images':arrays[key+'_images'][j],'labels':label,'split':0 if scene<24 else 1 if scene<32 else 2,'episode':key,'frame':step,'episode_length':121,'scene':scene,'contact_mode':mode,'mass_level':mi,'force_level':fi,'actual_states':arrays[key+'_fullstates'][step],'time_seconds':step*dt}
    for k,v in row.items():dataset[k].append(v)
  dataset={k:np.asarray(v) for k,v in dataset.items()};dataset['target_names']=np.array(['mass_kg','lateral_friction','force_N']);dataset['state_names']=np.array(['x','y','z','qx','qy','qz','qw','vx','vy','vz','wx','wy','wz']);assert len(dataset['images'])==a.scenes*18*len(STEPS)
  np.savez_compressed(a.output/'dataset.npz',**dataset);np.savez_compressed(a.output/'full_trajectories_with_repeats.npz',**arrays);dump(a.output/'records.json',records);dump(a.output/'scene_metadata.json',scene_meta);dump(a.output/'validation.json',{'status':'PASS_INVARIANTS','scenes':a.scenes,'frames':len(dataset['images']),'unique_learning_branches':a.scenes*18,'excluded_baseline_repeats':a.scenes*2,'checks':checks,'initial_ON_OFF_images_and_states_exact':True,'OFF_robot_force_zero_tolerance':1e-8,'floor_positive_force_every_branch':True,'every_step_contact_log':True,'dataset_sha256':hashlib.sha256((a.output/'dataset.npz').read_bytes()).hexdigest()})
 except Exception as ex:
  if arrays:np.savez_compressed(a.output/'partial_trajectories.npz',**arrays)
  dump(a.output/'FAILED.json',{'error':str(ex),'traceback':traceback.format_exc(),'records':records,'scene_metadata':scene_meta,'checks':checks});raise
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
