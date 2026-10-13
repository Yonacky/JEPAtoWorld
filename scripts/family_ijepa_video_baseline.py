"""G7d: frozen I-JEPA per-frame features, same 16 observations and video crop."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from complex_spatial_features import extract, sha, array_sha, device_metadata, check_features
from family_video_adapter import episode_indices, COMMIT


def video_crop(image):
    """Exact official ndarray geometry: OpenCV uint8 bilinear + rounded center."""
    import cv2
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
    h, w = image.shape[:2]
    nh, nw = (int(292*h/w), 292) if w < h else (292, int(292*w/h))
    x = image if min(h, w) == 292 else cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0, x0 = int(round((nh-256)/2)), int(round((nw-256)/2))
    return np.ascontiguousarray(x[y0:y0+256, x0:x0+256])


class CroppedFrames:
    """Only materialize the current batch, avoiding a 2.3GB resized frame copy."""
    def __init__(self, images): self.images = images
    def __len__(self): return len(self.images)
    def __getitem__(self, key):
        if isinstance(key, slice): return np.stack([video_crop(x) for x in self.images[key]])
        return video_crop(self.images[key])


def mapping(data):
    groups = episode_indices(data, 16)
    rows = np.stack([ids for _, ids in groups])
    np.testing.assert_array_equal(np.sort(rows.ravel()), np.arange(len(data['episode'])))
    for scene in np.unique(data['scene']):
        assert len(np.unique(data['split'][data['scene'] == scene])) == 1, 'scene split leakage'
    first = rows[:, 0]
    keys = ['scene','contact_mode','mass_level','force_level','split','episode','labels']
    manifest = {k:data[k][first] for k in keys}
    branchkeys = list(zip(*[manifest[k].tolist() for k in ['scene','contact_mode','mass_level','force_level']]))
    assert len(set(branchkeys)) == len(branchkeys), 'duplicate branch'
    manifest.update(source_rows=rows, clip_frames=data['frame'][rows],
                    frame=data['frame'][first], episode_length=np.full(len(first),16),
                    target_names=data['target_names'])
    return rows, manifest


def aggregate(mean, grid, rows):
    check_features(mean, grid, len(mean))
    branch = mean[rows].mean(axis=1, dtype=np.float32)
    pooled = grid[rows].mean(axis=1, dtype=np.float32)
    check_features(branch, pooled, len(rows))
    return {'branch_vector':branch, 'grid':pooled, 'temporal_mean':mean[rows]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['dataset','model-root','root','output']: p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='auto');p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args();assert a.batch_size>0
    import torch, cv2
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    sys.path.insert(0,str(a.root/'core/e03'))
    from device_utils import activate_device
    device=activate_device(a.device)
    with np.load(a.dataset,allow_pickle=False) as z:
        data={k:z[k] for k in ['images','scene','contact_mode','mass_level','force_level','split','episode','frame','labels','target_names','time_seconds']}
    assert data['images'].dtype==np.uint8 and data['images'].shape[-1]==3
    rows,manifest=mapping(data)
    assert rows.shape==(720,16), 'frozen production requires 40 scenes x 18 branches'
    a.output.mkdir(parents=True,exist_ok=False)
    mean,grid,validation=extract(CroppedFrames(data['images']),a.model_root,device,a.batch_size)
    features=aggregate(mean,grid,rows)
    np.savez_compressed(a.output/'features.npz',**features)
    np.savez_compressed(a.output/'sample_manifest.npz',**manifest)
    vendor=a.model_root/'vendor/ijepa-main/src'
    config=dict(dataset_sha256=sha(a.dataset),images_sha256=array_sha(data['images']),
      checkpoint_sha256=sha(a.model_root/'models/ijepa_vith14_1k/model.safetensors'),
      code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('complex_spatial_features.py'),Path(__file__).with_name('family_video_adapter.py')]},
      vendor_sha256={str(p.relative_to(vendor)):sha(p) for p in sorted(vendor.rglob('*.py'))},
      geometry_reference_commit=COMMIT,opencv=cv2.__version__,torch=torch.__version__,
      geometry='ndarray uint8 cv2 INTER_LINEAR shortedge292 then round center crop256; no float-to-uint8 roundtrip',
      ijepa_preprocessing='cropped uint8 /255, FP32 bilinear antialias224 align_cornersFalse, (x-.5)/.5',
      precision='FP32; TF32 disabled; no autocast',pooling='per-frame spatial mean1280 then equal time mean over16; grid5120 likewise',
      limitation='same observations and geometric field of view; final224 vs video256, interpolation, normalization, architecture and pretraining differ; order discarded by time mean',
      args={k:str(v) for k,v in vars(a).items()},**device_metadata(torch,device),
      extraction_validation=validation)
    window=dict(task='retrospective_readout',sample_steps=list(range(0,121,8)),
      pooling='per-frame spatial mean1280 then time mean16',encoder_id='ijepa_vith14_1k_same_video_crop_time_mean',
      primary_feature_key='branch_vector',source_dataset_sha256=config['dataset_sha256'])
    for name,obj in [('config.json',config),('window_spec.json',window),('COMPLETE.json',{'status':'PASS','branches':720,'frames':11520,'mapping_partition':True})]:
        (a.output/name).write_text(json.dumps(obj,indent=2),encoding='utf8')
    (a.output/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in a.output.iterdir() if p.is_file()},indent=2))


if __name__=='__main__': main()
