"""G-3 fixed demo-disjoint extraction. Task-local evaluation; never states alignment."""
import argparse,pathlib,json,os,time,hashlib,traceback
import numpy as np
from complex_libero_pilot import extract,sha,dump,validate
SEED=20261012

def prepare(path,stride=4):
 import h5py
 out={k:[] for k in ['images','labels','actions','actions_valid','split','episode','frame','episode_length','ee_displacement','ee_displacement_valid','next_frame']};episodes=[]
 with h5py.File(path,'r') as f:
  names=sorted([k for k in f['data'] if k.startswith('demo_')],key=lambda k:int(k.rsplit('_',1)[-1]));assert len(names)>=20,('need 20 demos',len(names))
  selected=np.random.default_rng(SEED).permutation(names)[:20].tolist()
  for j,name in enumerate(selected):
   group=f['data'][name];obs=group['obs'];n=len(obs['agentview_rgb']);ids=np.arange(0,n,stride);assert len(ids)>=2,('short episode',name,n)
   assert len(obs['ee_pos'])==len(obs['gripper_states'])==n
   im=obs['agentview_rgb'][ids];assert im.shape[1:]==(128,128,3) and im.dtype==np.uint8
   ee=obs['ee_pos'][ids].astype(np.float32);gr=obs['gripper_states'][ids].astype(np.float32);labels=np.concatenate([ee,gr],axis=1);assert labels.shape[1]==5
   assert 'actions' in group,('missing actions',name);action=group['actions'];assert action.ndim==2;valid=ids<len(action);ac=np.full((len(ids),action.shape[1]),np.nan,dtype=np.float32);ac[valid]=action[ids[valid]]
   # Only same-obs sampled endpoint differences, no simulator state or velocity claim.
   delta=np.full_like(ee,np.nan);delta[:-1]=ee[1:]-ee[:-1];dv=np.arange(len(ids))<len(ids)-1;nextfr=np.full(len(ids),-1,dtype=np.int64);nextfr[:-1]=ids[1:]
   split=0 if j<12 else 1 if j<16 else 2
   values={'images':im[:,::-1,:,:].copy(),'labels':labels,'actions':ac,'actions_valid':valid,'split':np.full(len(ids),split,np.int8),'episode':np.repeat(name,len(ids)),'frame':ids,'episode_length':np.full(len(ids),n),'ee_displacement':delta,'ee_displacement_valid':dv,'next_frame':nextfr}
   for k,v in values.items():out[k].append(v)
   episodes.append({'demo':name,'split':split,'obs_rows':n,'action_rows':len(action),'sampled_frames':len(ids),'valid_action_rows':int(valid.sum()),'hdf5_obs_path':f'data/{name}/obs','hdf5_action_path':f'data/{name}/actions'})
 data={k:np.concatenate(v) for k,v in out.items()};data['target_names']=np.array(['ee_pos_x','ee_pos_y','ee_pos_z','gripper_joint_0','gripper_joint_1']);data['action_names']=np.array([f'action_{j}' for j in range(data['actions'].shape[1])]);data['displacement_names']=np.array(['delta_ee_x','delta_ee_y','delta_ee_z']);validate(data)
 assert np.isfinite(data['actions'][data['actions_valid']]).all();assert np.isfinite(data['ee_displacement'][data['ee_displacement_valid']]).all()
 meta={'selected_demos':selected,'unused_demo_count':len(names)-20,'episodes':episodes,'split_demo_counts':[12,4,4],'split_frame_counts':[int(sum(data['split']==s)) for s in range(3)],'seed':SEED,'stride':stride,'labels':'same obs row ee_pos/gripper_states','actions':'stored actions row at same integer index only; action-to-observation causal timing NOT verified; invalid/missing final action marked NaN and mask','displacement':'next sampled obs ee_pos minus current within SAME demo; endpoint frames explicit; not velocity or physical dynamics','states':'not read, no alignment assumption','orientation':'one fixed vertical flip matching complex_libero_pilot LIBERO OpenGL convention'}
 return data,meta

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--hdf5',type=pathlib.Path,nargs='+',required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--model-root',type=pathlib.Path,required=True);ap.add_argument('--stride',type=int,default=4);ap.add_argument('--batch-size',type=int,default=16);args=ap.parse_args();assert args.stride==4,'G3 frozen stride4'
 import torch
 torch.set_num_threads(4)
 if torch.cuda.is_available():
  assert torch.cuda.device_count()==1,'Externally restrict CUDA_VISIBLE_DEVICES to the ONE authorized GPU UUID before running';device=torch.device('cuda:0')
 else:device=torch.device('cpu')
 assert len(set(p.stem for p in args.hdf5))==len(args.hdf5),'duplicate task output names'
 args.output.mkdir(parents=True,exist_ok=False) # never overwrite failure or completed artifacts
 config={'seed':SEED,'hdf5':[str(p) for p in args.hdf5],'model_root':str(args.model_root),'stride':4,'split':[12,4,4],'batch':args.batch_size,'device':str(device),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'device_name':torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU','source_sha256':sha(pathlib.Path(__file__)),'extract_dependency_sha256':sha(pathlib.Path(__file__).with_name('complex_libero_pilot.py')),'scope':'each task evaluated separately; new task development, not proof of cross-task training generalization','checkpoint_sha256':sha(args.model_root/'models/ijepa_vith14_1k/model.safetensors')}
 dump(args.output/'config.json',config);start=time.time();tasks=[]
 try:
  for hdf in args.hdf5:
   td=args.output/hdf.stem;td.mkdir(exist_ok=False);data,meta=prepare(hdf,args.stride);meta['hdf5_source_sha256']=sha(hdf);meta['hdf5_path']=str(hdf);dump(td/'data.json',meta)
   np.savez_compressed(td/'sample_manifest.npz',**{k:v for k,v in data.items() if k!='images'})
   from PIL import Image
   Image.fromarray(data['images'][0]).save(td/'first_upright_frame.png')
   feats,exmeta=extract(data['images'],args.model_root,device,args.batch_size,td)
   assert feats.shape==(len(data['frame']),1280) and feats.dtype==np.float32
   audit={'mean_sha256':hashlib.sha256(feats.tobytes()).hexdigest(),'manifest_sha256':sha(td/'sample_manifest.npz'),'features_file_sha256':sha(td/'features.npz'),'source_hash_matches':sha(hdf)==meta['hdf5_source_sha256'],'demo_counts':meta['split_demo_counts'],'frames':len(feats),'finite':bool(np.isfinite(feats).all()),'feature_shape':list(feats.shape),'images_hash_matches':exmeta['images_sha256']==hashlib.sha256(data['images'].tobytes()).hexdigest()};assert audit['source_hash_matches'] and audit['images_hash_matches'];dump(td/'validation.json',audit);dump(td/'COMPLETE.json',{'status':'extraction and sample audit complete; downstream route evaluation pending'});tasks.append({'task':hdf.stem,'frames':len(feats),'validation':audit});del feats,data
  dump(args.output/'COMPLETE.json',{'status':'extraction only complete; route comparison pending','seconds':time.time()-start,'tasks':tasks})
 except Exception as e:
  dump(args.output/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'completed_tasks':tasks,'seconds':time.time()-start});raise
if __name__=='__main__':main()
