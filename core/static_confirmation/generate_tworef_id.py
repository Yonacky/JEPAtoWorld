"""Static factorial renderer benchmark, not a legacy trajectory replay."""
from pathlib import Path
import argparse,json,itertools,hashlib,copy
import numpy as np,pybullet as pb,gym
from causal_world.envs import CausalWorld
from causal_world.configs.world_constants import WorldConstants
from causal_world.task_generators.task import generate_task
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--scenes',type=int,default=48);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);assert not (a.output/'dataset.npz').exists()
rng=np.random.default_rng(20261131);levels=[np.array([-.075,0.,.075]),np.array([.2,.5,.8]),np.array([.2,.5,.8])];comb=list(itertools.product(range(3),repeat=3));n=a.scenes
scene_split=np.full(n,2,dtype=np.int8);assert np.all(scene_split==2)
config={'seed':20261131,'scenes':n,'factors':['object_x','object_green','floor_tint'],'levels':[v.tolist() for v in levels],'variants_per_scene':27,'scene_split':scene_split.tolist(),'rendering':'public CausalWorld pushing fixture, camera0, direct stage setters; no simulation steps after initialization','purpose':'new controlled renderer interventions; not exact BISCUIT replay; no dynamics identifiability claim','state_restore':'explicit reapplication of all controlled factors; audited ABA reversibility, not env.set_state across interventions','gym':gym.__version__,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(a.output/'config_before_render.json').write_text(json.dumps(config,indent=2))
e=CausalWorld(task=generate_task(task_generator_id='pushing'),observation_mode='pixel',enable_visualization=False,seed=20261131);e.add_ground_truth_state_to_info();e.reset()
original_step=pb.stepSimulation
steps=0
def forbidden(*args,**kwargs):
 global steps
 steps+=1;raise RuntimeError('Physics advanced during matched render')
pb.stepSimulation=forbidden

def flat(obj,prefix=''):
 d={}
 for k,v in obj.items():
  name=prefix+k
  if isinstance(v,dict):d.update(flat(v,name+'/'))
  else:d[name]=np.asarray(v).copy()
 return d

def snapshot():
 e._robot.update_latest_full_state()
 return flat(e._task.get_info()['ground_truth_current_state_variables'])

def render():
 before=snapshot();im=e._robot.get_current_camera_observations()[0];after=snapshot()
 for k in before:np.testing.assert_allclose(before[k],after[k],rtol=0,atol=1e-12)
 assert im.shape==(128,128,3) and np.isfinite(im).all() and im.min()>=0 and im.max()<=1
 return np.rint(im*255).astype(np.uint8)

def set_factors(c,y,yaw):
 x,g,b=[levels[j][c[j]] for j in range(3)]
 e._stage.apply_interventions({'floor_color':np.array([.1,.1,.1])+np.array([.45,.45,.9])*b,'tool_block':{'cartesian_position':np.array([x,y,.0325]),'euler_orientation':np.array([0.,0.,yaw]),'color':np.array([.5,g,.5]),'linear_velocity':np.zeros(3),'angular_velocity':np.zeros(3)}})
 s=snapshot();np.testing.assert_allclose(s['tool_block/cartesian_position'],[x,y,.0325],atol=1e-10);np.testing.assert_allclose(s['tool_block/color'],[.5,g,.5],atol=1e-7)
 np.testing.assert_allclose(s['floor_color'],[.1+.45*b,.1+.45*b,.1+.9*b],atol=1e-7)
 # Read actual renderer geometry/color and physics, not only internal task labels.
 obj=e._stage.get_rigid_objects()['tool_block'];contacts=[]
 for client,body in zip(obj._pybullet_client_ids,obj._block_ids):
  np.testing.assert_allclose(pb.getBasePositionAndOrientation(body,physicsClientId=client)[0],[x,y,.0325+WorldConstants.FLOOR_HEIGHT],atol=1e-10)
  visual=pb.getVisualShapeData(body,physicsClientId=client);np.testing.assert_allclose(visual[0][7][:3],[.5,g,.5],atol=1e-7)
  dyn=pb.getDynamicsInfo(body,-1,physicsClientId=client);assert abs(dyn[0]-.02)<1e-10 and abs(dyn[1]-.175)<1e-10
  pb.performCollisionDetection(physicsClientId=client)
  cp=pb.getContactPoints(bodyA=body,physicsClientId=client);assert all(v[8]>-1e-5 for v in cp),('penetration',cp)
  contacts.append(len(cp))
 return s,contacts
images=[];states=[];scene_ids=[];codes=[];labels=[];checks=[];contexts=[];allpairs=[]
try:
 for scene in range(n):
  y=float(rng.uniform(-.035,.035));yaw=float(rng.uniform(-.5,.5));stage=float(rng.uniform(.55,.95));joints=np.tile([0.,np.pi/4,-np.pi/4],3);joints[[0,3,6]]=rng.uniform(-.12,.12,3)
  e._robot.apply_interventions({'joint_positions':joints,'joint_velocities':np.zeros(9)})
  e._stage.apply_interventions({'stage_color':np.repeat(stage,3),'floor_friction':.2,'stage_friction':.3,'tool_block':{'friction':.175,'mass':.02}})
  e._stage.apply_interventions({'tool_block':{'friction':.175}}) # legacy setter uses elif for mass/friction, so set separately
  contexts.append({'scene':scene,'y':y,'yaw':yaw,'stage_gray':stage,'joints':joints.tolist()})
  start=len(images);baseline=None;reference=None
  for c in comb:
   s,contacts=set_factors(c,y,yaw);im=render();assert np.array_equal(im,render()),'repeat render nondeterministic'
   if baseline is None:baseline=s;reference=im.copy()
   for k in s:
    if k not in ['tool_block/cartesian_position','tool_block/cylindrical_position','tool_block/color','floor_color']:np.testing.assert_allclose(s[k],baseline[k],rtol=0,atol=1e-10,err_msg=k)
   images.append(im);states.append(s);scene_ids.append(scene);codes.append(c);labels.append([levels[j][c[j]] for j in range(3)])
  set_factors(comb[0],y,yaw);assert np.array_equal(render(),reference),'ABA restoration failed'
  changed=[]
  for i,c in enumerate(comb):
   for factor in range(3):
    for val in range(c[factor]+1,3):
     d=list(c);d[factor]=val;j=comb.index(tuple(d));s1,s2=states[start+i],states[start+j]
     allow=[['tool_block/cartesian_position','tool_block/cylindrical_position'],['tool_block/color'],['floor_color']][factor]
     for k in s1:
      if k not in allow:np.testing.assert_allclose(s1[k],s2[k],rtol=0,atol=1e-10,err_msg=k)
     pixels=int(np.any(images[start+i]!=images[start+j],axis=-1).sum());assert pixels>=5,('invisible intervention',scene,c,d,pixels)
     allpairs.append([start+i,start+j,factor,pixels]);changed.append(pixels)
  checks.append({'scene':scene,'ABA_exact':True,'repeat_exact':True,'non_target_state_invariant':True,'minimum_changed_pixels':min(changed)})
  print('scene',scene,'frames',len(images),'min_pixels',min(changed),flush=True)
finally:
 pb.stepSimulation=original_step;e.close()
keys=sorted(states[0]);offsets=np.cumsum([0]+[states[0][k].size for k in keys]);vectors=np.array([np.concatenate([s[k].reshape(-1) for k in keys]) for s in states],dtype=np.float64)
np.savez_compressed(a.output/'dataset.npz',images=np.array(images),labels=np.array(labels),scene=np.array(scene_ids),split=scene_split[np.array(scene_ids)],codes=np.array(codes),pairs=np.array(allpairs),state_values=vectors,state_keys=np.array(keys),state_offsets=offsets,target_names=np.array(config['factors']))
from PIL import Image
sheet=Image.new('RGB',(128*9,128*3))
for row in range(3):
 for col in range(9):sheet.paste(Image.fromarray(images[row*9+col]),(col*128,row*128))
sheet.save(a.output/'scene0_grid.png')
(a.output/'contexts.json').write_text(json.dumps(contexts,indent=2))
result={'status':'PASS','scenes':n,'frames':len(images),'single_factor_pairs':len(allpairs),'simulation_step_attempts':steps,'checks':checks,'dataset_sha256':hashlib.sha256((a.output/'dataset.npz').read_bytes()).hexdigest()}
(a.output/'validation.json').write_text(json.dumps(result,indent=2));print('COMPLETE',len(images),len(allpairs),flush=True)
