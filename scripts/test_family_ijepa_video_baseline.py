"""CPU synthetic mapping and pinned official crop equivalence; no encoder call."""
import ast, json, sys, types, numbers
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parents[1]/'runs/family_ijepa_video_baseline_validation_20261013'
sys.path.insert(0,str(O/'deps'))
import cv2, torch, PIL.Image
from family_ijepa_video_baseline import video_crop,mapping,aggregate,CroppedFrames,sha


def official_parts(path,names,ns):
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)


ns=dict(np=np,cv2=cv2,torch=torch,PIL=PIL,numbers=numbers)
official_parts(O/'functional.py',['resize_clip','get_resize_sizes','crop_clip'],ns)
ns['FF']=types.SimpleNamespace(**{k:ns[k] for k in ['resize_clip','crop_clip']})
official_parts(O/'transforms.py',['Resize','CenterCrop'],ns)
rng=np.random.default_rng(20261013);checks=[]
for h,w in [(128,128),(127,191),(193,128),(292,333),(333,292),(256,257)]:
    im=rng.integers(0,256,(h,w,3),dtype=np.uint8)
    reference=ns['CenterCrop']((256,256))(ns['Resize'](292,interpolation='bilinear')([im]))[0]
    actual=video_crop(im);np.testing.assert_array_equal(actual,reference)
    np.testing.assert_array_equal(CroppedFrames(np.stack([im,im]))[:2],np.stack([reference,reference]))
    checks.append({'shape':[h,w],'maxabs':0,'dtype':str(actual.dtype)})
scene=np.repeat(np.arange(40),18*16);branch=np.repeat(np.arange(720),16)
d=dict(scene=scene,episode=branch,split=np.where(scene<24,0,np.where(scene<32,1,2)),
       contact_mode=np.where((branch//9)%2==0,'on','off'),mass_level=(branch%9)//3,
       force_level=branch%3,frame=np.tile(np.arange(0,121,8),720),labels=np.ones((11520,3)),
       target_names=np.array(['mass','friction','force']))
d['time_seconds']=d['frame']*.004
order=rng.permutation(11520);d={k:(v[order] if len(v)==11520 else v) for k,v in d.items()}
rows,m=mapping(d)
np.testing.assert_array_equal(d['frame'][rows],np.tile(np.arange(0,121,8),(720,1)))
np.testing.assert_array_equal(d['episode'][rows],np.broadcast_to(m['episode'][:,None],(720,16)))
rejected=[]
for kind in ['duplicate_frame','split_change','scene_leakage']:
    bad={k:v.copy() for k,v in d.items()};ii=rows[0]
    if kind=='duplicate_frame':bad['frame'][ii[1]]=bad['frame'][ii[0]]
    elif kind=='split_change':bad['split'][ii[1]]=(int(bad['split'][ii[1]])+1)%3
    else:bad['split'][ii]=(int(bad['split'][ii[0]])+1)%3
    try: mapping(bad)
    except (AssertionError,ValueError):rejected.append(kind)
    else: raise AssertionError('invalid data accepted: '+kind)
grid=rng.normal(size=(32,5120)).astype(np.float32)
mean=grid.reshape(32,1280,4).mean(2)
rr=np.arange(32).reshape(2,16)[:,::-1]
features=aggregate(mean,grid,rr)
np.testing.assert_allclose(features['branch_vector'],np.stack([mean[x].astype(float).mean(0) for x in rr]),atol=1e-7)
result=dict(status='PASS_CPU_ONLY',seed=20261013,official_crop_checks=checks,
            full_shuffled_mapping=[720,16],invalid_inputs_rejected=rejected,
            aggregation='FP32 time mean matches independent float64 reduction atol1e-7',
            gpu_or_checkpoint_executed=False,opencv=cv2.__version__)
(O/'validation.json').write_text(json.dumps(result,indent=2))
(O/'manifest.json').write_text(json.dumps({str(p):sha(p) for p in [O/'validation.json',O/'functional.py',O/'transforms.py',Path(__file__),Path(__file__).with_name('family_ijepa_video_baseline.py')]},indent=2))
print(json.dumps(result))
