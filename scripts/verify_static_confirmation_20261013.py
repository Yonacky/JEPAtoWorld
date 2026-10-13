"""Independent G7i raw static-data audit; does not trust producer validation flags."""
import argparse,pathlib,json,hashlib,itertools,numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def context_vector(c):return np.array([c['y'],c['yaw'],c['stage_gray'],*c['joints']],float)
def planned_contexts(seed,scenes,background):
 rng=np.random.default_rng(seed);out=[]
 for i in range(scenes):
  y=float(rng.uniform(-.035,.035));yaw=float(rng.uniform(-.5,.5));stage=float(rng.uniform(*background));j=np.tile([0.,np.pi/4,-np.pi/4],3);j[[0,3,6]]=rng.uniform(-.12,.12,3);out.append({'scene':i,'y':y,'yaw':yaw,'stage_gray':stage,'joints':j.tolist()})
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--scenes',type=int,default=64);p.add_argument('--background',nargs=2,type=float,required=True);p.add_argument('--old-contexts',type=pathlib.Path,nargs='+',required=True);p.add_argument('--peer-contexts',type=pathlib.Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 source=a.data;cfg=json.loads((source/'config_before_render.json').read_text());launch=json.loads((source/'launch_config_before_render.json').read_text());ctx=json.loads((source/'contexts.json').read_text());n=a.scenes
 assert cfg['seed']==a.seed and cfg['scenes']==n and launch['seed']==a.seed and launch['scenes']==n
 assert launch['background_range']==a.background and not launch['prepare_only']
 pre=json.loads((source/'pre_generation_hashes.json').read_text())
 for f,h in pre.items():assert sha(source/f)==h,('pre_generation_hash_mismatch',f)
 assert sha(source/'effective_generator.py')==cfg['source_sha256']==launch['effective_source_sha256']
 with np.load(source/'dataset.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
 with np.load(source/'audit_raw.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
 im=d['images'];code=d['codes'];scene=d['scene'];S=d['state_values'];keys=d['state_keys'].tolist();offset=d['state_offsets'];comb=np.array(list(itertools.product(range(3),repeat=3)));levels=np.array([[-.075,0,.075],[.2,.5,.8],[.2,.5,.8]])
 assert im.dtype==np.uint8 and im.shape==(n*27,128,128,3);assert S.shape[0]==n*27 and np.isfinite(S).all();assert np.array_equal(np.unique(scene),np.arange(n));assert np.all(d['split']==2)
 assert len(ctx)==n and [c['scene'] for c in ctx]==list(range(n));planned=planned_contexts(a.seed,n,a.background);np.testing.assert_array_equal(np.array([context_vector(c) for c in ctx]),np.array([context_vector(c) for c in planned]))
 assert keys==raw['state_keys'].tolist() and np.array_equal(offset,raw['state_offsets'])
 fields={k:S[:,offset[i]:offset[i+1]] for i,k in enumerate(keys)}
 np.testing.assert_array_equal(raw['repeat_images'],im)
 assert raw['render_before'].shape==raw['render_after'].shape==(n*55,S.shape[1]);render_error=float(np.abs(raw['render_before']-raw['render_after']).max());assert render_error<=1e-12
 pair_rows=[];counts=np.zeros((n,3),int);mins=np.full((n,3),128*128,int);invariant_error=0.;aba_state_error=0.;render_source_error=0.
 allow=[{'tool_block/cartesian_position','tool_block/cylindrical_position'},{'tool_block/color'},{'floor_color'}]
 for sc in range(n):
  ids=np.flatnonzero(scene==sc);assert len(ids)==27;np.testing.assert_array_equal(code[ids],comb);np.testing.assert_array_equal(d['labels'][ids],np.stack([levels[k,comb[:,k]] for k in range(3)],1))
  np.testing.assert_array_equal(raw['aba_images'][sc],im[ids[0]]);ae=float(np.abs(raw['aba_state_values'][sc]-S[ids[0]]).max());aba_state_error=max(aba_state_error,ae);assert ae<=1e-10
  rb=raw['render_before'][sc*55:(sc+1)*55];rs=np.concatenate([np.repeat(S[ids],2,axis=0),S[ids[:1]]],axis=0);re=float(np.abs(rb-rs).max());render_source_error=max(render_source_error,re);assert re<=1e-10
  np.testing.assert_allclose(fields['tool_block/cartesian_position'][ids],np.column_stack([levels[0,comb[:,0]],np.repeat(ctx[sc]['y'],27),np.repeat(.0325,27)]),atol=1e-10,rtol=0)
  np.testing.assert_allclose(fields['tool_block/color'][ids],np.column_stack([np.repeat(.5,27),levels[1,comb[:,1]],np.repeat(.5,27)]),atol=1e-7,rtol=0)
  np.testing.assert_allclose(fields['floor_color'][ids],.1+levels[2,comb[:,2],None]*np.array([.45,.45,.9]),atol=1e-7,rtol=0)
  np.testing.assert_allclose(fields['stage_color'][ids],np.full((27,3),ctx[sc]['stage_gray']),atol=1e-7,rtol=0)
  np.testing.assert_allclose(fields['joint_positions'][ids],np.repeat(np.array(ctx[sc]['joints'])[None],27,axis=0),atol=1e-10,rtol=0)
  for zero in ['joint_velocities','tool_block/linear_velocity','tool_block/angular_velocity']:np.testing.assert_allclose(fields[zero][ids],0,rtol=0,atol=1e-10)
  for i,c in enumerate(comb):
   for k in range(3):
    for value in range(c[k]+1,3):
     dest=c.copy();dest[k]=value;j=int(dest@[9,3,1]);aidx,bidx=ids[i],ids[j]
     for field,values in fields.items():
      if field not in allow[k]:
       err=float(np.abs(values[aidx]-values[bidx]).max());invariant_error=max(invariant_error,err);assert err<=1e-10,(sc,k,field,err)
     pixels=int(np.any(im[aidx]!=im[bidx],axis=-1).sum());assert pixels>=5,(sc,k,pixels);counts[sc,k]+=1;mins[sc,k]=min(mins[sc,k],pixels);pair_rows.append([int(aidx),int(bidx),k,pixels])
 np.testing.assert_array_equal(d['pairs'],np.array(pair_rows));assert np.all(counts==27)
 for name,value in [('tool_block/mass',.02),('tool_block/friction',.175),('floor_friction',.2),('stage_friction',.3)]:np.testing.assert_allclose(fields[name],value,atol=1e-10,rtol=0)
 bullets=json.loads((source/'actual_bullet.json').read_text());clients=sorted(set(b['client'] for b in bullets));assert len(bullets)==n*28*len(clients)
 tally={}
 for b in bullets:
  assert 0<=b['scene']<n and len(b['codes'])==3;assert abs(b['mass']-.02)<1e-10 and abs(b['friction']-.175)<1e-10
  np.testing.assert_allclose(b['rgba'][:3],[.5,levels[1,b['codes'][1]],.5],atol=1e-7,rtol=0)
  assert b['minimum_contact_distance'] is None or b['minimum_contact_distance']>-1e-5
  key=(b['scene'],b['client'],*b['codes']);tally[key]=tally.get(key,0)+1
 for sc,client,c in itertools.product(range(n),clients,comb.tolist()):assert tally[(sc,client,*c)]==(2 if c==[0,0,0] else 1)
 # Context overlap independently checked against every explicitly supplied prior source.
 cv=np.array([context_vector(c) for c in ctx]);assert len(np.unique(cv,axis=0))==n;old_checks=[]
 for pth in a.old_contexts+([a.peer_contexts] if a.peer_contexts else []):
  previous=json.loads(pth.read_text());pv=np.array([context_vector(c) for c in previous]);delta=np.max(np.abs(cv[:,None]-pv[None]),axis=2);hit=np.argwhere(delta<=1e-12)
  assert len(hit)==0,('context_collision',str(pth),hit.tolist())
  old_checks.append({'path':str(pth),'sha256':sha(pth),'scenes':len(previous),'matching_within_1e12':len(hit),'nearest_maxnorm_context_difference':float(delta.min())})
 with np.load(source/'prepared.npz',allow_pickle=False) as z:
  for k in ['images','labels','split','target_names']:np.testing.assert_array_equal(z[k],d[k])
  np.testing.assert_array_equal(z['episode'],scene);np.testing.assert_array_equal(z['frame'],code@np.array([9,3,1]));assert np.all(z['episode_length']==27)
 result={'status':'PASS_SAVED_DATA_AUDIT','seed':a.seed,'scenes':n,'frames':len(im),'single_factor_pairs':len(pair_rows),'two_reference_indices_per_scene':[0,26],'simulation_steps':'not independently reconstructed from saved data; monkeypatch guard checked in producer source','render_state_maxabs':render_error,'render_to_saved_state_maxabs':render_source_error,'aba_state_maxabs':aba_state_error,'repeat_pixels_exact':True,'aba_pixels_exact':True,'nontarget_state_maxabs':invariant_error,'minimum_changed_pixels_by_factor':mins.min(0).tolist(),'actual_bullet_records':len(bullets),'old_context_checks':old_checks,'peer_context_checked':a.peer_contexts is not None,'prepared_rows_exact':True,'dataset_sha256':sha(source/'dataset.npz'),'audit_raw_sha256':sha(source/'audit_raw.npz'),'audit_source_sha256':sha(pathlib.Path(__file__))}
 (a.output/'validation.json').write_text(json.dumps(result,indent=2));np.savez_compressed(a.output/'per_scene_visibility.npz',minimum_changed_pixels=mins,pair_counts=counts)
 (a.output/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in a.output.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in result.items() if k!='old_context_checks'},indent=2))
if __name__=='__main__':main()
