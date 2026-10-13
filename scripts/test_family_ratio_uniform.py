"""Uniform branch source mapping regression checks; synthetic CPU only."""
from pathlib import Path
import numpy as np,json,subprocess,sys,hashlib
O=Path('runs/family_ratio_readout_uniform_validation_20261013');O.mkdir(exist_ok=False);old=Path('runs/family_ratio_readout_validation_20261013');d=np.load(old/'dataset.npz');ix=np.flatnonzero(d['frame']==0);keys=['scene','contact_mode','split','episode','mass_level','force_level','labels'];steps=np.arange(0,121,8);md={k:np.repeat(d[k][ix],16,axis=0) for k in keys};md['frame']=np.tile(steps,720);md['time_seconds']=md['frame']*.004;np.savez_compressed(O/'dataset.npz',**md);rng=np.random.default_rng(20261013);order=np.arange(719,-1,-1);rows=np.arange(11520).reshape(720,16)[order];fm={k:md[k][rows[:,0]] for k in keys};fm.update(source_rows=rows,clip_frames=md['frame'][rows]);np.savez_compressed(O/'features.npz',branch_vector=rng.normal(size=(720,9)));np.savez_compressed(O/'sample_manifest.npz',**fm);(O/'window.json').write_text(json.dumps({'task':'retrospective_readout','sample_steps':steps.tolist(),'pooling':'synthetic','encoder_id':'synthetic_uniform_fixture'}));checks=[]
def run(name,manifest,success):
 cmd=[sys.executable,'scripts/family_ratio_readout.py','--data',str(O/'dataset.npz'),'--features',str(O/'features.npz'),'--output',str(O/name),'--input-kind','branch_vector','--dataset-manifest',str(manifest),'--window-spec',str(O/'window.json')];r=subprocess.run(cmd,capture_output=True,text=True);(O/(name+'.log')).write_text(r.stdout+r.stderr);assert (r.returncode==0)==success,r.stderr;checks.append({'case':name,'exit_code':r.returncode,'expected_success':success})
run('valid_uniform',O/'sample_manifest.npz',True)
bad={k:v.copy() for k,v in fm.items()};bad['source_rows'][0,0]=bad['source_rows'][1,0];np.savez_compressed(O/'bad_source.npz',**bad);run('reject_source_rows',O/'bad_source.npz',False)
bad={k:v.copy() for k,v in fm.items()};bad['clip_frames'][0,1]=9;np.savez_compressed(O/'bad_frames.npz',**bad);run('reject_clip_frames',O/'bad_frames.npz',False)
cmd=[sys.executable,'scripts/family_ratio_readout.py','--data',str(old/'dataset.npz'),'--features',str(old/'image_d7/features.npz'),'--output',str(O/'image_d7')];r=subprocess.run(cmd,capture_output=True,text=True);assert r.returncode==0,r.stderr;(O/'image_d7.log').write_text(r.stdout+r.stderr)
for mode in ['on','off']:
 for target in ['logratio','logmass','logforce']:
  a=np.load(O/'image_d7'/f'{mode}_{target}.npz');b=np.load(old/'image_d7/readout'/f'{mode}_{target}.npz')
  for k in a.files:np.testing.assert_array_equal(a[k],b[k])
checks.append({'case':'image_d7_historical_synthetic','six_models_all_saved_fields':'exact'})
(O/'validation.json').write_text(json.dumps({'status':'PASS','checks':checks,'source_sha256':hashlib.sha256(Path('scripts/family_ratio_readout.py').read_bytes()).hexdigest(),'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'not_encoder_results':True},indent=2));print(checks)
