"""G7i: thin audited wrapper around frozen static factorial generator; no physics rewrite.
Requires original generate_tworef_id.py passed explicitly. --prepare-only compiles/saves
source and config without importing CausalWorld. Each invocation needs a fresh output.
"""
import argparse, pathlib, hashlib, json, sys, traceback
BASE_SHA='f8f4ba66c319ca50c74ee66819f11390ff9725afa5658b746324ca2710b99e81'
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def replace_once(s,old,new):
 if s.count(old)!=1:raise ValueError('Legacy source drift: '+old[:100])
 return s.replace(old,new)
def prepare(legacy,output,seed,scenes,background):
 raw=legacy.read_bytes();s=raw.decode('utf-8').replace('\r\n','\n')
 if sha_bytes(s.encode())!=BASE_SHA:raise ValueError('Unexpected legacy generator SHA256')
 original=s
 parser=next(l for l in s.splitlines() if l.startswith('p=argparse.ArgumentParser();'))
 s=replace_once(s,parser,"from types import SimpleNamespace\na=SimpleNamespace(output=Path("+repr(str(output))+"),scenes="+str(scenes)+")")
 if s.count('20261131')!=3:raise ValueError('Unexpected seed substitutions')
 s=s.replace('20261131',str(seed))
 s=replace_once(s,'stage=float(rng.uniform(.55,.95))',f'stage=float(rng.uniform({background[0]!r},{background[1]!r}))')
 # Pure audit storage: no new physics steps, RNG draws, interventions or renders.
 s=replace_once(s,'images=[];states=[];', 'audit_repeats=[];audit_aba_images=[];audit_aba_states=[];audit_before=[];audit_after=[];audit_bullet=[]\nimages=[];states=[];')
 s=replace_once(s,'before=snapshot();im=e._robot.get_current_camera_observations()[0];after=snapshot()', 'before=snapshot();im=e._robot.get_current_camera_observations()[0];after=snapshot();audit_before.append(before);audit_after.append(after)')
 s=replace_once(s,"s,contacts=set_factors(c,y,yaw);im=render();assert np.array_equal(im,render()),'repeat render nondeterministic'", "s,contacts=set_factors(c,y,yaw);im=render();repeat_im=render();audit_repeats.append(repeat_im);assert np.array_equal(im,repeat_im),'repeat render nondeterministic'")
 s=replace_once(s,"set_factors(comb[0],y,yaw);assert np.array_equal(render(),reference),'ABA restoration failed'", "aba_s,_=set_factors(comb[0],y,yaw);aba_im=render();audit_aba_images.append(aba_im);audit_aba_states.append(aba_s);assert np.array_equal(aba_im,reference),'ABA restoration failed'")
 s=replace_once(s,'contacts.append(len(cp))',"audit_bullet.append({'scene':scene,'codes':list(c),'client':client,'body':body,'mass':float(dyn[0]),'friction':float(dyn[1]),'inertia':list(dyn[2]),'rgba':list(visual[0][7]),'minimum_contact_distance':min([float(v[8]) for v in cp],default=None),'contact_count':len(cp)})\n  contacts.append(len(cp))")
 s+='''\n# G7i added persisted audits; all arrays were captured from pre-existing operations.
def audit_vectors(rows):return np.array([np.concatenate([s[k].reshape(-1) for k in keys]) for s in rows],dtype=np.float64)
np.savez_compressed(a.output/'audit_raw.npz',repeat_images=np.array(audit_repeats),aba_images=np.array(audit_aba_images),aba_state_values=audit_vectors(audit_aba_states),render_before=audit_vectors(audit_before),render_after=audit_vectors(audit_after),state_keys=np.array(keys),state_offsets=offsets)
(a.output/'actual_bullet.json').write_text(json.dumps(audit_bullet,indent=2))
np.savez_compressed(a.output/'prepared.npz',images=np.array(images),labels=np.array(labels),split=scene_split[np.array(scene_ids)],episode=np.array(scene_ids),frame=np.array(codes)@np.array([9,3,1]),episode_length=np.full(len(images),27,dtype=np.int32),target_names=np.array(config['factors']))
'''
 compile(s,str(output/'effective_generator.py'),'exec')
 return raw,s

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-script',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--scenes',type=int,default=64);p.add_argument('--background',nargs=2,type=float,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 if a.scenes<=0:raise ValueError('scenes must be positive')
 if not 0<=a.background[0]<a.background[1]<=1:raise ValueError('Invalid background range')
 a.output=a.output.resolve();raw,source=prepare(a.legacy_script,a.output,a.seed,a.scenes,a.background)
 a.output.mkdir(parents=True,exist_ok=False)
 effective=a.output/'effective_generator.py';effective.write_text(source,encoding='utf-8');(a.output/'legacy_generator_snapshot.py').write_bytes(raw)
 cfg={'seed':a.seed,'scenes':a.scenes,'background_range':a.background,'scene_split':'all2 evaluation only; no training or tuning from these scenes','legacy_source_sha256':sha_bytes(raw),'legacy_normalized_sha256':BASE_SHA,'effective_source_sha256':sha_bytes(source.encode()),'wrapper_sha256':sha_bytes(pathlib.Path(__file__).read_bytes()),'changes':['CLI seed/scenes/background/output','persist existing repeat/ABA images and state','persist render-before/after states and Bullet dynamics/visual values'],'unchanged':'original27factor grid, all context ranges except explicit background, RNG draw order, one env, zero stepping, old setters incl mass/friction separation, camera, rendering, pair logic','prepare_only':a.prepare_only}
 dump(a.output/'launch_config_before_render.json',cfg);dump(a.output/'pre_generation_hashes.json',{f.name:sha_bytes(f.read_bytes()) for f in [effective,a.output/'legacy_generator_snapshot.py',a.output/'launch_config_before_render.json']})
 if a.prepare_only:print('PREPARED_NOT_GENERATED',a.output);return
 try:
  exec(compile(source,str(effective),'exec'),{'__name__':'__main__','__file__':str(effective)})
  files=[f for f in a.output.iterdir() if f.is_file()];dump(a.output/'manifest.json',{f.name:sha_bytes(f.read_bytes()) for f in files})
  dump(a.output/'COMPLETE.json',{'status':'GENERATED_PENDING_INDEPENDENT_AUDIT','scenes':a.scenes,'frames':a.scenes*27,'seed':a.seed,'manifest_sha256':sha_bytes((a.output/'manifest.json').read_bytes())})
 except BaseException:
  dump(a.output/'FAILED.json',{'traceback':traceback.format_exc(),'launch_config_sha256':sha_bytes((a.output/'launch_config_before_render.json').read_bytes())});raise
if __name__=='__main__':main()
