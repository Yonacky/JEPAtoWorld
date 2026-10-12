"""S1: reuse old20 exactly, extract only remaining30, frozen old tune/dev."""
import argparse,pathlib,json,os,time,traceback,hashlib
import numpy as np
from complex_libero_pilot import extract,sha,dump
from extract_libero_multitask_20261012 import prepare,SEED

def extra_data(path,names):
 import h5py
 out={k:[] for k in ['images','labels','actions','actions_valid','split','episode','frame','episode_length','ee_displacement','ee_displacement_valid','next_frame']}
 with h5py.File(path,'r') as f:
  for name in names:
   g=f['data'][name];o=g['obs'];n=len(o['agentview_rgb']);ids=np.arange(0,n,4);assert len(ids)>=2;assert len(o['ee_pos'])==len(o['gripper_states'])==n
   im=o['agentview_rgb'][ids];assert im.dtype==np.uint8 and im.shape[1:]==(128,128,3)
   ee=o['ee_pos'][ids].astype(np.float32);gr=o['gripper_states'][ids].astype(np.float32);labels=np.concatenate([ee,gr],axis=1);assert labels.shape[1]==5
   action=g['actions'];valid=ids<len(action);ac=np.full((len(ids),action.shape[1]),np.nan,np.float32);ac[valid]=action[ids[valid]];dv=np.arange(len(ids))<len(ids)-1;delta=np.full_like(ee,np.nan);delta[:-1]=ee[1:]-ee[:-1];nf=np.full(len(ids),-1,np.int64);nf[:-1]=ids[1:]
   row={'images':im[:,::-1].copy(),'labels':labels,'actions':ac,'actions_valid':valid,'split':np.zeros(len(ids),np.int8),'episode':np.repeat(name,len(ids)),'frame':ids,'episode_length':np.full(len(ids),n),'ee_displacement':delta,'ee_displacement_valid':dv,'next_frame':nf}
   for k,v in row.items():out[k].append(v)
 return {k:np.concatenate(v) for k,v in out.items()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--hdf5',nargs='+',type=pathlib.Path,required=True);ap.add_argument('--old-root',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--model-root',type=pathlib.Path,required=True);ap.add_argument('--batch-size',type=int,default=16);a=ap.parse_args()
 import h5py,torch
 torch.set_num_threads(4);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');assert device.type=='cpu' or torch.cuda.device_count()==1,'Restrict to ONE authorized GPU UUID externally'
 a.output.mkdir(parents=True,exist_ok=False);start=time.time();done=[]
 config={'seed':SEED,'stride':4,'training_sizes':[12,24,42],'old_eval_reused_not_fresh_confirmation':True,'source_sha256':sha(pathlib.Path(__file__)),'extract_dependency_sha256':sha(pathlib.Path(__file__).with_name('complex_libero_pilot.py')),'prepare_dependency_sha256':sha(pathlib.Path(__file__).with_name('extract_libero_multitask_20261012.py')),'device':str(device),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')};dump(a.output/'config.json',config)
 try:
  for hdf in a.hdf5:
   old=a.old_root/hdf.stem;td=a.output/hdf.stem;td.mkdir(exist_ok=False);oldmeta=json.loads((old/'data.json').read_text());oldextract=json.loads((old/'features.json').read_text());sourcehash=sha(hdf);assert sourcehash==oldmeta['hdf5_source_sha256'],'HDF5 source changed';assert sha(a.model_root/'models/ijepa_vith14_1k/model.safetensors')==oldextract['checkpoint_sha256'];assert oldextract['precision']=='FP32, TF32 disabled'
   with h5py.File(hdf,'r') as f:names=sorted([k for k in f['data'] if k.startswith('demo_')],key=lambda k:int(k.rsplit('_',1)[-1]))
   assert len(names)==50,('expected exactly50',len(names));order=np.random.default_rng(SEED).permutation(names).tolist();assert order[:20]==oldmeta['selected_demos'];trainorder=order[:12]+order[20:];assert len(trainorder)==42
   # Reload raw old20 ONLY to validate source/manifest/image hash; NEVER recompute their model features.
   check,_=prepare(hdf,4);manifest=dict(np.load(old/'sample_manifest.npz'));oldfeat=np.load(old/'features.npz')['mean'];assert oldfeat.dtype==np.float32 and oldfeat.shape==(len(check['frame']),1280);assert np.isfinite(oldfeat).all()
   for k,v in manifest.items():np.testing.assert_equal(v,check[k])
   assert hashlib.sha256(check['images'].tobytes()).hexdigest()==oldextract['images_sha256'];del check
   added=extra_data(hdf,order[20:]);newdir=td/'new30_extraction';newdir.mkdir();newfeat,newmeta=extract(added['images'],a.model_root,device,a.batch_size,newdir)
   for k in ['checkpoint_sha256','precision','resize','normalization','pooling']:assert newmeta[k]==oldextract[k],('preprocessing mismatch',k)
   combined={k:np.concatenate([v,added[k]]) if k in added and k!='images' else v for k,v in manifest.items()};features=np.concatenate([oldfeat,newfeat]);assert features.dtype==np.float32;combined['train_order_index']=np.array([trainorder.index(ep) if ep in trainorder else -1 for ep in combined['episode']],np.int16)
   np.savez_compressed(td/'features.npz',mean=features);np.savez_compressed(td/'sample_manifest.npz',**combined)
   splits={str(n):trainorder[:n] for n in [12,24,42]};dump(td/'train_order.json',{'training_order':trainorder,'nested_training_demos':splits,'tune_demos':order[12:16],'dev_demos':order[16:20],'permutation50':order,'storage_order':'old20 rows unchanged prefix, followed by new30 train rows'})
   counts=[len(np.unique(combined['episode'][combined['split']==s])) for s in range(3)];assert counts==[42,4,4]
   loaded=np.load(td/'features.npz')['mean'];np.testing.assert_array_equal(loaded[:len(oldfeat)],oldfeat)
   reloaded=np.load(td/'sample_manifest.npz')
   for k,v in manifest.items():np.testing.assert_equal(reloaded[k][:len(v)] if k in added else reloaded[k],v)
   audit={'passed':True,'old20_features_bitwise_unchanged':True,'all_old_manifest_fields_unchanged':True,'old_frames':len(oldfeat),'new_frames':len(newfeat),'total_frames':len(features),'split_demo_counts':counts,'source_hash_unchanged':sha(hdf)==sourcehash,'finite_features':bool(np.isfinite(features).all()),'features_sha256':sha(td/'features.npz'),'manifest_sha256':sha(td/'sample_manifest.npz'),'old_features_sha256':sha(old/'features.npz')};assert audit['source_hash_unchanged'] and audit['finite_features'];dump(td/'validation.json',audit)
   dump(td/'data.json',{**oldmeta,'selected_demos':order,'split_demo_counts':counts,'unused_demo_count':0,'episodes_note':'inherited episodes describe old20 only; full50 source identity in sample_manifest, permutation50 and train_order.json','training_order':trainorder,'old_root':str(old),'hdf5_source_sha256':sourcehash,'nested_training_sizes':[12,24,42],'reuse':'old20 untouched; new30 only feature inference','status':'old tune/dev reused exploratory learning curve'})
   dump(td/'features.json',{'precision':newmeta['precision'],'checkpoint_sha256':newmeta['checkpoint_sha256'],'pooling':newmeta['pooling'],'old20_metadata':str(old/'features.json'),'new30_metadata':'new30_extraction/features.json','combined_features_not_reencoded':True});dump(td/'COMPLETE.json',audit);done.append(hdf.stem)
  dump(a.output/'COMPLETE.json',{'tasks':done,'seconds':time.time()-start,'status':'extraction complete; nested evaluations pending'})
 except Exception as e:dump(a.output/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'completed_tasks':done});raise
if __name__=='__main__':main()
