import pathlib,sys,numpy as np,json,hashlib,h5py
sys.path.insert(0,'F:/2610/CVPR27/scripts');import extend_libero_learning_curve_20261012 as m
r=pathlib.Path('F:/2610/CVPR27/runs/libero_learning_curve_preparation_20261012');r.mkdir(exist_ok=True);hdf=r/'fixture.hdf5'
with h5py.File(hdf,'w') as f:
 for k in range(50):
  g=f.create_group(f'data/demo_{k}');o=g.create_group('obs');o.create_dataset('agentview_rgb',data=np.full((9,128,128,3),k,np.uint8),compression='gzip');o['ee_pos']=np.arange(27).reshape(9,3)+k*.1;o['gripper_states']=np.arange(18).reshape(9,2)+k*.1;g['actions']=np.ones((8,7))*k
model=r/'model';ck=model/'models/ijepa_vith14_1k/model.safetensors';ck.parent.mkdir(parents=True,exist_ok=True);ck.write_bytes(b'synthetic fixture only')
oldroot=r/'old';old=oldroot/'fixture';old.mkdir(parents=True,exist_ok=True);data,meta=m.prepare(hdf);meta['hdf5_source_sha256']=m.sha(hdf);m.dump(old/'data.json',meta);np.savez(old/'sample_manifest.npz',**{k:v for k,v in data.items() if k!='images'})
calls=[]
def fake(images,model_root,device,batch,out):
 calls.append(len(images));features=np.repeat(images[:,0,0,:1].astype(np.float32),1280,axis=1);em={'images_sha256':hashlib.sha256(images.tobytes()).hexdigest(),'checkpoint_sha256':m.sha(ck),'precision':'FP32, TF32 disabled','resize':'fixed','normalization':'fixed','pooling':'mean1280'};np.savez(out/'features.npz',mean=features);m.dump(out/'features.json',em);return features,em
fake(data['images'],model,None,16,old);calls.clear();m.extract=fake
sys.argv=['mock','--hdf5',str(hdf),'--old-root',str(oldroot),'--output',str(r/'combined'),'--model-root',str(model)];m.main();assert calls==[90],calls
v=json.loads((r/'combined/fixture/validation.json').read_text());assert v['old_frames']==60 and v['new_frames']==90;m.dump(r/'loader_validation.json',{'passed':True,'mock_feature_inference_rows':calls,'old20_reused_exactly':v['old20_features_bitwise_unchanged'],'manifest_unchanged':v['all_old_manifest_fields_unchanged'],'split_demo_counts':v['split_demo_counts'],'scope':'synthetic50 end-to-end data plumbing; real model inference not tested','source_sha256':m.sha(pathlib.Path(m.__file__))});print('PASS only90 new rows, old60 exact,42/4/4')
