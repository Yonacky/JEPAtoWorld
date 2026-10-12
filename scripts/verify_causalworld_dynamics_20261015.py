"""Independent G5a file audit. Recompute from arrays/records, never trust checks flags."""
import pathlib,argparse,json,hashlib,numpy as np

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((a.data/'config.json').read_text());records=json.loads((a.data/'records.json').read_text());z=np.load(a.data/'dataset.npz',allow_pickle=False);full=np.load(a.data/'full_trajectories_with_repeats.npz',allow_pickle=False);lookup={r['key']:r for r in records};assert len(lookup)==len(records)==320
 assert cfg['seed']==20261015;steps=np.array([0,1,5,10,20,30,60,90,120]);assert np.array_equal(cfg['record_steps'],steps);clients=[v['client'] for v in cfg['clients']];dt=np.array([v['physics']['fixedTimeStep'] for v in cfg['clients']]);assert np.max(abs(dt-.004))<1e-12
 assert z['images'].shape==(2520,128,128,3) and z['images'].dtype==np.uint8;assert np.isfinite(z['actual_states']).all() and np.isfinite(z['labels']).all();assert len(set(zip(z['episode'].tolist(),z['frame'].tolist())))==2520;assert set(z['scene'])==set(range(40));assert len(set(z['episode']))==280
 effects=[];repeat_errors=[];initial_errors=[];inertia_per_mass=[];dataset_keys=set(z['episode'].tolist());baseline_refs=[]
 for scene in range(40):
  rs=[r for r in records if r['scene']==scene];assert len(rs)==8;bl=[r for r in rs if r['factor']=='baseline' and r['repeat']==0];br=[r for r in rs if r['factor']=='baseline' and r['repeat']==1];assert len(bl)==len(br)==1;bk=bl[0]['key'];rk=br[0]['key'];assert bk in dataset_keys and rk not in dataset_keys
  repeaterr=float(np.max(abs(full[bk+'_fullstates']-full[rk+'_fullstates'])));repeat_errors.append(repeaterr);assert repeaterr<1e-10;np.testing.assert_array_equal(full[bk+'_images'],full[rk+'_images']);baseinitial=full[bk+'_fullstates'][0];baseimage=full[bk+'_images'][0]
  split=0 if scene<24 else 1 if scene<32 else 2;si=np.flatnonzero(z['scene']==scene);assert len(si)==63;assert np.all(z['split'][si]==split)
  for r in rs:
   key=r['key'];factor=r['factor'];lev=r['level'];mass=lev if factor=='mass' else .02;friction=lev if factor=='friction' else .175;force=lev if factor=='force' else .02;states=full[key+'_fullstates'];sampled=full[key+'_states'];images=full[key+'_images'];assert states.shape==(121,len(clients),13);assert sampled.shape==(9,len(clients),13);assert np.isfinite(states).all();np.testing.assert_array_equal(sampled,states[steps]);np.testing.assert_array_equal(images[0],baseimage);initialerr=float(np.max(abs(states[0]-baseinitial)));initial_errors.append(initialerr);assert initialerr<1e-12
   # Quaternion and start velocity checks are recomputed, not copied from producer checks.
   np.testing.assert_allclose(np.linalg.norm(states[:,:,3:7],axis=-1),1,atol=1e-8);np.testing.assert_allclose(states[0,:,7:],0,atol=1e-12);assert r['force_n']==force;assert r['sample_steps']==steps.tolist();assert len(r['contact_counts'])==8
   assert [d['client'] for d in r['actual_dynamics']]==clients
   for d in r['actual_dynamics']:
    assert abs(d['mass']-mass)<1e-12 and abs(d['lateral_friction']-friction)<1e-12;inertia=np.array(d['local_inertia']);assert np.all(inertia>0);inertia_per_mass.append(inertia/mass)
   if r['repeat']==0:
    idx=np.flatnonzero(z['episode']==key);assert len(idx)==9;np.testing.assert_array_equal(z['frame'][idx],steps);np.testing.assert_array_equal(z['images'][idx],images);np.testing.assert_array_equal(z['actual_states'][idx],sampled);np.testing.assert_allclose(z['labels'][idx],np.tile([mass,friction,force],(9,1)),rtol=0,atol=0);assert np.all(z['episode_length'][idx]==121);np.testing.assert_allclose(z['time_seconds'][idx],steps*dt[0],rtol=0,atol=1e-14);assert np.all(z['factor'][idx]==factor);expected_level=1 if factor=='baseline' else (0 if lev==cfg[factor+'_levels'][0] else 2);assert np.all(z['level'][idx]==expected_level)
  for factor in ['mass','friction','force']:
   rr=sorted([r for r in rs if r['factor']==factor],key=lambda r:r['level']);assert len(rr)==2;assert [r['level'] for r in rr]==[cfg[factor+'_levels'][0],cfg[factor+'_levels'][2]];lo,hi=[r['key'] for r in rr];posdiff=float(np.linalg.norm(full[lo+'_fullstates'][:,:,:3]-full[hi+'_fullstates'][:,:,:3],axis=-1).max());pixels=int(np.any(full[lo+'_images'][-1]!=full[hi+'_images'][-1],axis=-1).sum());assert posdiff>1e-4 and pixels>=5;effects.append({'scene':scene,'factor':factor,'max_future_position_difference':posdiff,'final_changed_pixels':pixels});baseline_refs.append({'scene':scene,'factor':factor,'low':lo,'mid':bk,'high':hi,'split':split})
 normalized=np.array(inertia_per_mass);drift=float(np.max(abs(normalized-normalized[0])));assert drift<1e-10,'unexpected normalized inertia drift across scenes/branches'
 result={'status':'PASS','recomputed_scenes':40,'learning_branches':280,'repeat_branches_excluded':40,'frames':2520,'scene_split':[24,8,8],'initial_state_maxabs':max(initial_errors),'repeat_state_maxabs':max(repeat_errors),'initial_and_repeat_images_exact':True,'normalized_inertia_across_all_branches_maxdrift':drift,'physics_clients':clients,'effects':effects,'scope':'Independent saved-data audit; no new simulation. DynamicsInfo records support true configured values, external force actual command timing verified in source not separately sensor-logged. Fullstates contains tool body only, not complete world.'}
 (a.output/'validation.json').write_text(json.dumps(result,indent=2));(a.output/'balanced_factor_episode_references.json').write_text(json.dumps(baseline_refs,indent=2))
 manifest={n:sha(a.data/n) for n in ['dataset.npz','full_trajectories_with_repeats.npz','records.json','config.json']};manifest['auditor_source_sha256']=sha(pathlib.Path(__file__));(a.output/'source_hashes.json').write_text(json.dumps(manifest,indent=2));print('PASS independently recomputed40 scenes280 branches2520 frames',flush=True)
if __name__=='__main__':main()
