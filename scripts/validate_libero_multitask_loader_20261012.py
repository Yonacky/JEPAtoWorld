import sys,pathlib,numpy as np,json
sys.path.insert(0,'F:/2610/CVPR27/scripts')
from extract_libero_multitask_20261012 import prepare,sha
import h5py
out=pathlib.Path('F:/2610/CVPR27/runs/libero_multitask_preparation_20261012');out.mkdir(exist_ok=True);p=out/'synthetic_loader_fixture.hdf5'
with h5py.File(p,'w') as f:
 for k in range(22):
  g=f.create_group(f'data/demo_{k}');o=g.create_group('obs');images=np.zeros((9,128,128,3),np.uint8);images[:,0,:,1]=127
  o.create_dataset('agentview_rgb',data=images,compression='gzip');o['ee_pos']=np.arange(27).reshape(9,3)+k*.1;o['gripper_states']=np.arange(18).reshape(9,2)+k*.1;g['actions']=np.ones((8,7))*k
x,meta=prepare(p);y,meta2=prepare(p);assert meta==meta2
for key in x:np.testing.assert_equal(x[key],y[key])
assert meta['split_demo_counts']==[12,4,4];assert x['images'].shape==(60,128,128,3);assert x['actions_valid'].sum()==40;assert x['ee_displacement_valid'].sum()==40
assert np.all(x['images'][:,-1,:,1]==127);np.testing.assert_allclose(x['ee_displacement'][x['ee_displacement_valid']],12,atol=2e-6)
for ep in np.unique(x['episode']):
 ids=np.flatnonzero(x['episode']==ep);assert np.array_equal(x['frame'][ids],[0,4,8]);assert np.array_equal(x['next_frame'][ids],[4,8,-1]);assert len(set(x['split'][ids]))==1
v={'passed':True,'deterministic_selection':True,'split_demo_counts':meta['split_demo_counts'],'frames':60,'last_action_masked_correctly':True,'no_cross_demo_displacement':True,'vertical_flip_verified':True,'source_sha256':sha(pathlib.Path('F:/2610/CVPR27/scripts/extract_libero_multitask_20261012.py')),'scope':'synthetic loader audit only; real HDF5/model extraction root remote'};(out/'loader_validation.json').write_text(json.dumps(v,indent=2));print(v)
