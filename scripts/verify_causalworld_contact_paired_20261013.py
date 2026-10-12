"""G5f independent array/record checks; load each compressed image array once."""
import pathlib,argparse,json,hashlib,numpy as np

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((a.data/'config.json').read_text());records=json.loads((a.data/'records.json').read_text());metadata=json.loads((a.data/'scene_metadata.json').read_text());n=cfg['scenes'];assert (n,cfg['seed']) in [(2,20261017),(40,20261018)]
 with np.load(a.data/'dataset.npz') as zz:z={k:zz[k] for k in zz.files}
 full=np.load(a.data/'full_trajectories_with_repeats.npz');assert len(records)==20*n and len(metadata)==n;assert len(set(r['key'] for r in records))==20*n;assert z['images'].shape==(n*162,128,128,3) and z['images'].dtype==np.uint8;assert len(set(z['episode']))==18*n;assert len(set(zip(z['episode'].tolist(),z['frame'].tolist())))==n*162
 steps=np.array([0,1,5,10,20,30,60,90,120]);maxinitial=maxrepeat=maxrobotrepeat=maxinertia=0.;offforce=0.;summaries=[];onpositive=offpositive=0;floorbranches=0;mode_pairs=[];initialpenetration=[]
 for sc in range(n):
  sm=metadata[sc];assert sm['scene']==sc and sm['env_seed']==cfg['seed']+sc;maps=sm['runtime_body_mapping'];clients=[m['client'] for m in maps];baseinertia=np.array(sm['inertia_per_mass']);np.testing.assert_allclose(baseinertia,np.array([np.array(d[2])/d[0] for d in sm['inertia_source_dynamics_info']]),rtol=0,atol=1e-12)
  for m,ll in zip(maps,sm['filter_robot_links']):
   ids={v['id']:v for v in m['body_info']};assert m['robot'] in ids and m['tool'] in ids and m['floor'] in ids;assert ids[m['robot']]['num_joints']>0 and ids[m['tool']]['num_joints']==0;assert ll==list(range(-1,ids[m['robot']]['num_joints']));assert abs(m['physics']['fixedTimeStep']-.004)<1e-12
  world=np.array(sm['initial_world_body_states']);robot0=np.array(sm['initial_robot_joint_states']);rr=[r for r in records if r['scene']==sc];assert len(rr)==20;cache={}
  for r in rr:
   key=r['key'];cache[key]={suffix:full[key+'_'+suffix] for suffix in ['images','fullstates','robot_states']}
  reference=cache[next(r['key'] for r in rr if r['contact_mode']=='on' and r['mass_kg']==.02 and r['force_n']==.02 and r['repeat']==0)]
  for r in rr:
   key=r['key'];mode=r['contact_mode'];mass=r['mass_kg'];force=r['force_n'];q=cache[key];states=q['fullstates'];robot=q['robot_states'];ims=q['images'];assert states.shape==(121,len(clients),13) and ims.shape==(9,128,128,3);assert robot.shape[0]==121 and robot.shape[1]==len(clients) and robot.shape[-1]==2;assert np.isfinite(states).all() and np.isfinite(robot).all();np.testing.assert_array_equal(ims[0],reference['images'][0]);er=float(np.max(abs(states[0]-reference['fullstates'][0])));maxinitial=max(maxinitial,er);assert er<1e-12;np.testing.assert_allclose(robot[0],robot0,atol=1e-12,rtol=0)
   for ci,m in enumerate(maps):
    ww=world[(world[:,0]==m['client'])&(world[:,1]==m['tool'])];assert len(ww)==1;np.testing.assert_allclose(states[0,ci],ww[0,2:],atol=1e-12,rtol=0)
   np.testing.assert_allclose(states[0,:,7:],0,atol=1e-12);np.testing.assert_allclose(np.linalg.norm(states[:,:,3:7],axis=-1),1,atol=1e-8);assert r['explicit_filter_enable']==int(mode=='on');assert r['filter_robot_links']==sm['filter_robot_links'];assert [d['client'] for d in r['actual_dynamics']]==clients
   for ci,d in enumerate(r['actual_dynamics']):
    assert abs(d['mass']-mass)<1e-12 and abs(d['friction']-.175)<1e-12;er=float(np.max(abs(np.array(d['inertia'])-baseinertia[ci]*mass)));maxinertia=max(maxinertia,er);assert er<1e-12
   cc=r['contacts_every_step'];assert [t['step'] for t in cc]==list(range(121));maxrf=0.;floorpos=0;initialdist=[]
   for t in cc:
    assert [c['client'] for c in t['clients']]==clients
    for ci,c in enumerate(t['clients']):
     for p in c['contacts']:
      assert p['bodyA']==maps[ci]['tool'];assert np.isfinite([p['normal_force'],p['lateral_force1'],p['lateral_force2'],p['distance']]).all();assert p['normal_force']>=-1e-10
      for k in ['position_on_A','position_on_B','normal_on_B','lateral_direction1','lateral_direction2']:assert len(p[k])==3 and np.isfinite(p[k]).all()
      if p['bodyB']==maps[ci]['robot']:
       maxrf=max(maxrf,abs(p['normal_force']))
       if t['step']==0:initialdist.append(p['distance'])
      if t['step']>0 and p['bodyB']==maps[ci]['floor'] and p['normal_force']>1e-8:floorpos+=1
   if mode=='off':offforce=max(offforce,maxrf);assert maxrf<=1e-8
   assert floorpos>0
   if r['repeat']==0:
    floorbranches+=1;onpositive+=int(mode=='on' and maxrf>1e-8);offpositive+=int(mode=='off' and maxrf>1e-8);idx=np.flatnonzero(z['episode']==key);assert len(idx)==9;np.testing.assert_array_equal(z['frame'][idx],steps);np.testing.assert_array_equal(z['images'][idx],ims);np.testing.assert_array_equal(z['actual_states'][idx],states[steps]);np.testing.assert_allclose(z['labels'][idx],np.tile([mass,.175,force],(9,1)),rtol=0,atol=0);assert np.all(z['scene'][idx]==sc) and np.all(z['contact_mode'][idx]==mode);assert np.all(z['split'][idx]==(0 if sc<24 else 1 if sc<32 else 2));assert np.all(z['mass_level'][idx]==[.01,.02,.04].index(mass)) and np.all(z['force_level'][idx]==[.01,.02,.03].index(force));np.testing.assert_allclose(z['time_seconds'][idx],steps*.004,rtol=0,atol=1e-14);assert np.all(z['episode_length'][idx]==121)
    if initialdist:initialpenetration.append(min(initialdist))
   else:
    assert key not in set(z['episode']);assert mass==force==.02;bk=next(x['key'] for x in rr if x['repeat']==0 and x['contact_mode']==mode and x['mass_kg']==mass and x['force_n']==force);b=cache[bk];er=float(np.max(abs(states-b['fullstates'])));re=float(np.max(abs(robot-b['robot_states'])));maxrepeat=max(maxrepeat,er);maxrobotrepeat=max(maxrobotrepeat,re);assert max(er,re)<1e-10;np.testing.assert_array_equal(ims,b['images'])
  for mass in [.01,.02,.04]:
   for force in [.01,.02,.03]:
    keys=[next(r['key'] for r in rr if r['contact_mode']==mode and r['mass_kg']==mass and r['force_n']==force and r['repeat']==0) for mode in ['on','off']];on,off=[cache[k] for k in keys];np.testing.assert_array_equal(on['images'][0],off['images'][0]);np.testing.assert_allclose(on['robot_states'][0],off['robot_states'][0],atol=1e-12,rtol=0);mode_pairs.append({'scene':sc,'mass':mass,'force':force,'max_tool_position_difference':float(np.linalg.norm(on['fullstates'][:,:,:3]-off['fullstates'][:,:,:3],axis=-1).max()),'max_robot_joint_position_difference':float(np.max(abs(on['robot_states'][...,0]-off['robot_states'][...,0]))),'final_image_changed_pixels':int(np.any(on['images'][-1]!=off['images'][-1],axis=-1).sum())})
  del cache
 result={'status':'PASS_INVARIANTS','scenes':n,'frames':n*162,'learning_branches':n*18,'excluded_repeats':n*2,'max_initial_tool_state_error':maxinitial,'max_repeat_tool_state_error':maxrepeat,'max_repeat_robot_state_error':maxrobotrepeat,'max_inertia_error':maxinertia,'OFF_max_robot_normal_force':offforce,'ON_branches_robot_force_gt_1e_8':onpositive,'OFF_branches_robot_force_gt_1e_8':offpositive,'branches_with_positive_floor_force':floorbranches,'initial_robot_contact_distance_range':[min(initialpenetration),max(initialpenetration)] if initialpenetration else None,'paired_mode_differences':mode_pairs,'limitations':['Collision enable is explicit command metadata; OFF effect checked using every-step returned contact force.','Initial all-body states saved in scene metadata; branch-level saved trajectories verify tool and robot joints only.','Initial penetration may be released by collision removal; all steps retained.','Robot later motion can differ despite identical initial control settings.']}
 (a.output/'validation.json').write_text(json.dumps(result,indent=2));(a.output/'source_hashes.json').write_text(json.dumps({**{n:sha(a.data/n) for n in ['dataset.npz','full_trajectories_with_repeats.npz','records.json','scene_metadata.json','config.json']},'auditor_source_sha256':sha(pathlib.Path(__file__))},indent=2));print('PASS',n,'scenes OFFforce',offforce,'ONpositive',onpositive,flush=True)
if __name__=='__main__':main()
