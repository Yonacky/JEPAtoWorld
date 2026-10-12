"""Independent crossed saved-data audit, no producer validation/check flags read."""
import pathlib,argparse,json,hashlib,numpy as np

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((a.data/'config.json').read_text());records=json.loads((a.data/'records.json').read_text());z=np.load(a.data/'dataset.npz');full=np.load(a.data/'full_trajectories_with_repeats.npz');assert len(records)==400 and len(set(r['key'] for r in records))==400;assert cfg['seed']==20261016;steps=np.array([0,1,5,10,20,30,60,90,120]);clients=[c['client'] for c in cfg['clients']];templates=np.array(cfg['fixed_shape_inertia_per_mass']);raw=cfg['inertia_template_source_dynamics_info'];np.testing.assert_allclose(templates,np.array([np.array(d[2])/d[0] for d in raw]),rtol=0,atol=1e-12)
 assert z['images'].shape==(3240,128,128,3) and z['images'].dtype==np.uint8;assert len(set(z['episode']))==360;assert len(set(zip(z['episode'].tolist(),z['frame'].tolist())))==3240;assert np.isfinite(z['actual_states']).all() and np.isfinite(z['labels']).all()
 axes=[];ratios=[];repe=[];initial=[];contact_bodies=set();maxinertia=0.;dt=cfg['clients'][0]['physics']['fixedTimeStep'];assert abs(dt-.004)<1e-12
 for scene in range(40):
  rr=[r for r in records if r['scene']==scene];assert len(rr)==10;grid={(r['mass_kg'],r['force_n']):r for r in rr if r['repeat']==0};assert set(grid)==set((m,f) for m in [.01,.02,.04] for f in [.01,.02,.03]);baseline=grid[(.02,.02)]['key'];repeat=[r for r in rr if r['repeat']==1];assert len(repeat)==1 and repeat[0]['mass_kg']==.02 and repeat[0]['force_n']==.02;rk=repeat[0]['key'];assert rk not in set(z['episode']);err=float(np.max(abs(full[baseline+'_fullstates']-full[rk+'_fullstates'])));assert err<1e-10;repe.append(err);np.testing.assert_array_equal(full[baseline+'_images'],full[rk+'_images'])
  idxscene=np.flatnonzero(z['scene']==scene);assert len(idxscene)==81;split=0 if scene<24 else 1 if scene<32 else 2;assert np.all(z['split'][idxscene]==split)
  for r in rr:
   key=r['key'];mass=r['mass_kg'];force=r['force_n'];ss=full[key+'_fullstates'];sampled=full[key+'_states'];im=full[key+'_images'];assert ss.shape==(121,len(clients),13);assert np.isfinite(ss).all();np.testing.assert_array_equal(sampled,ss[steps]);np.testing.assert_array_equal(im[0],full[baseline+'_images'][0]);er=float(np.max(abs(ss[0]-full[baseline+'_fullstates'][0])));initial.append(er);assert er<1e-12;np.testing.assert_allclose(ss[0,:,7:],0,atol=1e-12);np.testing.assert_allclose(np.linalg.norm(ss[:,:,3:7],axis=-1),1,atol=1e-8)
   assert [d['client'] for d in r['actual_dynamics']]==clients
   for ci,d in enumerate(r['actual_dynamics']):
    assert abs(d['mass']-mass)<1e-12 and abs(d['lateral_friction']-.175)<1e-12;e=float(np.max(abs(np.array(d['local_inertia'])-templates[ci]*mass)));maxinertia=max(maxinertia,e);assert e<1e-12
   assert [c['step'] for c in r['contact_details']]==steps[1:].tolist()
   for details,counts in zip(r['contact_details'],r['contact_counts']):
    assert [d['client'] for d in details['clients']]==clients
    for d,count in zip(details['clients'],counts):
     assert len(d['contacts'])==count
     for c in d['contacts']:assert np.isfinite(c['normal_force']) and c['normal_force']>=-1e-10;contact_bodies.add(c['bodyB'])
   if r['repeat']==0:
    ids=np.flatnonzero(z['episode']==key);assert len(ids)==9;np.testing.assert_array_equal(z['frame'][ids],steps);np.testing.assert_array_equal(z['actual_states'][ids],sampled);np.testing.assert_array_equal(z['images'][ids],im);np.testing.assert_allclose(z['labels'][ids],np.tile([mass,.175,force],(9,1)),atol=0,rtol=0);assert np.all(z['mass_level'][ids]==[.01,.02,.04].index(mass)) and np.all(z['force_level'][ids]==[.01,.02,.03].index(force));assert np.all(z['episode_length'][ids]==121);np.testing.assert_allclose(z['time_seconds'][ids],steps*dt,atol=1e-14,rtol=0)
  def compare(pair):
   keys=[grid[p]['key'] for p in pair];diff=full[keys[0]+'_fullstates']-full[keys[1]+'_fullstates'];pixels=int(np.any(full[keys[0]+'_images'][-1]!=full[keys[1]+'_images'][-1],axis=-1).sum());return {'scene':scene,'pairs':pair,'position_maxdiff':float(np.linalg.norm(diff[:,:,:3],axis=-1).max()),'velocity_maxdiff':float(np.linalg.norm(diff[:,:,7:10],axis=-1).max()),'final_changed_pixels':pixels}
  for mass in [.01,.02,.04]:axes.append({'factor':'force',**compare([(mass,.01),(mass,.03)])})
  for force in [.01,.02,.03]:axes.append({'factor':'mass',**compare([(.01,force),(.04,force)])})
  for pair in [[(.01,.01),(.02,.02)],[(.02,.01),(.04,.02)]]:
   assert abs(pair[0][1]/pair[0][0]-pair[1][1]/pair[1][0])<1e-12;ratios.append(compare(pair))
 result={'status':'PASS_INVARIANTS','scenes':40,'frames':3240,'unique_learning_branches':360,'excluded_repeats':40,'scene_split':[24,8,8],'max_initial_state_error':max(initial),'max_repeat_state_error':max(repe),'initial_and_repeat_images_exact':True,'max_inertia_template_error':maxinertia,'contact_bodyB_ids':sorted(contact_bodies),'axis_effects':axes,'equal_force_mass_ratio_pairs':ratios,'weak_axis_effect_count':sum(x['position_maxdiff']<=1e-4 or x['final_changed_pixels']<5 for x in axes),'limitations':['No effect threshold used to reject nine-grid cells.','force_n is applied-command metadata; actual force timing audited in source, not independent force sensor.','Only tool trajectory is stored; not full world state.','Equal F/m is a diagnostic and not a required physical equivalence.']}
 (a.output/'validation.json').write_text(json.dumps(result,indent=2));(a.output/'source_hashes.json').write_text(json.dumps({**{n:sha(a.data/n) for n in ['dataset.npz','full_trajectories_with_repeats.npz','records.json','config.json']},'auditor_source_sha256':sha(pathlib.Path(__file__))},indent=2));print('PASS_INVARIANTS3240 frames360 branches; weak axis effects',result['weak_axis_effect_count'])
if __name__=='__main__':main()
