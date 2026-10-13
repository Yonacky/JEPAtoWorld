"""CPU-only recovery after features were saved but metadata construction failed.

Never invokes extract or loads encoder weights. Requires actual failed source/log.
"""
import argparse,json
from pathlib import Path
import numpy as np
from family_ijepa_video_baseline import mapping,sha,COMMIT
from complex_spatial_features import check_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['dataset','output','model-root','original-source','failure-log']:
        p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();o=a.output
    assert not (o/'COMPLETE.json').exists(),'Already complete; do not overwrite'
    files=[o/'features.npz',o/'sample_manifest.npz']
    before={f.name:sha(f) for f in files}
    original_hash=sha(a.original_source);log_hash=sha(a.failure_log)
    source=a.original_source.read_text(encoding='utf8')
    assert '**validation' in source and "pooling='per-frame" in source,'Expected failed adapter source'
    assert 'TypeError' in a.failure_log.read_text(encoding='utf8',errors='replace') and 'pooling' in a.failure_log.read_text(encoding='utf8',errors='replace'),'Expected failure log'
    with np.load(a.dataset,allow_pickle=False) as z:
        data={k:z[k] for k in ['scene','contact_mode','mass_level','force_level','split','episode','frame','labels','target_names','time_seconds']}
    rows,expected=mapping(data);assert rows.shape==(720,16)
    with np.load(files[1],allow_pickle=False) as fm:
        for k,v in expected.items():np.testing.assert_array_equal(fm[k],v,err_msg=k)
    with np.load(files[0],allow_pickle=False) as f:
        assert set(f.files)=={'branch_vector','grid','temporal_mean'}
        branch=f['branch_vector'];grid=f['grid'];temporal=f['temporal_mean']
    assert branch.dtype==grid.dtype==temporal.dtype==np.float32
    assert temporal.shape==(720,16,1280) and np.isfinite(temporal).all()
    consistency=check_features(branch,grid,720)
    temporal_error=float(abs(temporal.mean(1,dtype=np.float32)-branch).max())
    np.testing.assert_array_equal(temporal.mean(1,dtype=np.float32),branch)
    assert {f.name:sha(f) for f in files}==before,'Feature files changed during recovery'
    vendor=a.model_root/'vendor/ijepa-main/src';vendor_files=sorted(vendor.rglob('*.py'));assert vendor_files
    checkpoint=a.model_root/'models/ijepa_vith14_1k/model.safetensors';assert checkpoint.is_file()
    info=dict(recovered_metadata=True,original_adapter_sha256=original_hash,
      original_source_path=str(a.original_source),failure_log_sha256=log_hash,
      recovery_script_sha256=sha(Path(__file__)),dataset_sha256=sha(a.dataset),
      checkpoint_sha256=sha(checkpoint),vendor_sha256={str(f.relative_to(vendor)):sha(f) for f in vendor_files},
      existing_feature_hashes=before,geometry_reference_commit=COMMIT,
      geometry='ndarray uint8 cv2 INTER_LINEAR shortedge292, round center crop256',
      ijepa_preprocessing='cropped uint8 /255, FP32 bilinear antialias224 align_cornersFalse, (x-.5)/.5',
      pooling='per-frame spatial mean1280 then equal time mean over16; grid5120 likewise',
      precision_protocol='FP32; TF32 disabled; no autocast, per original source',
      extraction_validation={'status':'numeric values were not persisted before failure',
        'strict_load_control_flow':'original extract returned before features save; strict/all-key assertions are in original code; not rerun during recovery',
        'sdpa_attention_maxabs':None,'sdpa_full_model_maxabs':None},
      original_runtime={'device_uuid':None,'torch_version':None,'opencv_version':None,'peak_memory':None,
        'status':'not persisted here; consult separately saved root runtime logs; no fabricated values'},
      recovery_validation={'mapping_partition':True,'all_saved_metadata_exact':True,
        'finite_float32':True,'mean_grid_maxabs':consistency,'temporal_mean_maxabs':temporal_error,
        'saved_features_unchanged':True,'encoder_executed':False},
      limitation='same window and geometric field of view, but resolution/interpolation/normalization/pretraining differ; time mean discards order')
    window=dict(task='retrospective_readout',sample_steps=list(range(0,121,8)),
      pooling=info['pooling'],encoder_id='ijepa_vith14_1k_same_video_crop_time_mean',
      primary_feature_key='branch_vector',source_dataset_sha256=info['dataset_sha256'])
    for name,obj in [('config.json',info),('window_spec.json',window),('recovery_validation.json',info['recovery_validation'])]:
        assert not (o/name).exists(),f'Preserve existing {name}'
        (o/name).write_text(json.dumps(obj,indent=2),encoding='utf8')
    complete=dict(status='PASS_RECOVERED_METADATA',branches=720,frames=11520,
      recovered_after='post-save duplicate pooling keyword TypeError',encoder_reexecuted=False,
      original_adapter_sha256=original_hash,feature_hashes=before,
      missing_original_numeric_validation=True)
    (o/'COMPLETE.json').write_text(json.dumps(complete,indent=2),encoding='utf8')
    (o/'manifest.json').write_text(json.dumps({f.name:sha(f) for f in o.iterdir() if f.is_file() and f.name!='manifest.json'},indent=2),encoding='utf8')
    print(json.dumps(complete,indent=2))


if __name__=='__main__':main()
