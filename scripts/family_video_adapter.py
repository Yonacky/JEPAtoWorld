"""G7b pinned V-JEPA2 ViT-L encoder only. No download or physics changes.
Input dataset.npz: uint8 images[N,H,W,3], episode/frame and branch-constant metadata.
Outputs one vector per episode; whole-window features are NOT causal frame features.
"""
import argparse, hashlib, json, pathlib, subprocess, sys
import numpy as np

COMMIT = '204698b45b3712590f06245fbfba32d3be539812'
WEIGHT_URL = 'https://dl.fbaipublicfiles.com/vjepa2/vitl.pt'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def install_source(path):
    path=pathlib.Path(path).resolve()
    revision=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if revision!=COMMIT:raise ValueError(f'Expected source {COMMIT}, got {revision}')
    dirty=subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if dirty:raise ValueError('Pinned vendor has modified tracked files: '+dirty)
    if 'src' in sys.modules:raise RuntimeError('Use a fresh process: src namespace already imported')
    sys.path.insert(0,str(path))
    return path

def load_encoder(source,checkpoint,device):
    import torch
    source=install_source(source)
    from src.models.vision_transformer import vit_large
    # Match official hub defaults, retain dynamic actual T in the forward pass.
    model=vit_large(img_size=(256,256),patch_size=16,num_frames=64,tubelet_size=2,
                    use_sdpa=True,use_silu=False,wide_silu=True,uniform_power=False,
                    use_rope=True,handle_nonsquare_inputs=True)
    checkpoint=pathlib.Path(checkpoint)
    ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if 'target_encoder' not in ckpt:raise KeyError('Official standard pretrain requires target_encoder; no fallback')
    cleaned={}
    for k,v in ckpt['target_encoder'].items():
        new=k.replace('module.','').replace('backbone.','')
        if new in cleaned:raise ValueError('Duplicate cleaned key '+new)
        cleaned[new]=v
    removed=[]
    if 'pos_embed' in cleaned:
        p=cleaned.pop('pos_embed')
        if p.ndim!=3 or p.shape[0]!=1 or p.shape[-1]!=1024:raise ValueError('Unexpected legacy pos_embed shape')
        removed=[{'key':'pos_embed','shape':list(p.shape),'reason':'official RoPE path has no positional parameter'}]
    result=model.load_state_dict(cleaned,strict=True)
    del ckpt,cleaned
    model.eval().requires_grad_(False).to(device)
    return model,{'source_commit':COMMIT,'checkpoint_url':WEIGHT_URL,'checkpoint_sha256':sha(checkpoint),
                  'checkpoint_key':'target_encoder','removed_keys':removed,
                  'missing_keys':result.missing_keys,'unexpected_keys':result.unexpected_keys,
                  'parameter_count':sum(p.numel() for p in model.parameters()),
                  'source_files_sha256':{p:sha(source/p) for p in ['src/models/vision_transformer.py','src/models/utils/modules.py','src/models/utils/patch_embed.py','src/datasets/utils/video/transforms.py','src/datasets/utils/video/volume_transforms.py']}}

def preprocessor():
    # Exactly the official single-view evaluation transform, on uint8 RGB ndarray frames.
    from src.datasets.utils.video import transforms as vt, volume_transforms as vol
    return vt.Compose([vt.Resize(int(256*256/224),interpolation='bilinear'),
                       vt.CenterCrop(size=(256,256)),vol.ClipToTensor(),
                       vt.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225))])

def episode_indices(data,frames):
    ep=data['episode'];fr=data['frame'];groups=[]
    for e in dict.fromkeys(ep.tolist()):
        ids=np.flatnonzero(ep==e);ids=ids[np.argsort(fr[ids],kind='stable')]
        steps=fr[ids]
        if len(ids)!=frames or len(set(steps.tolist()))!=frames:raise ValueError(f'Bad frame count/duplicates: {e}')
        if not np.all(np.diff(steps)==np.diff(steps)[0]) or np.diff(steps)[0]<=0:raise ValueError('Nonuniform frames: '+str(e))
        for k in ['scene','split','contact_mode','mass_level','force_level','labels']:
            if k not in data:raise KeyError(k)
            if not np.all(data[k][ids]==data[k][ids[0]]):raise ValueError(f'Nonconstant {k} in {e}')
        if 'time_seconds' in data:
            np.testing.assert_allclose(data['time_seconds'][ids],steps*.004,rtol=0,atol=1e-12)
        groups.append((e,ids))
    if frames==16:
        for e,ids in groups:
            if not np.array_equal(fr[ids],np.arange(0,121,8)):raise ValueError('Frozen 16-frame steps must be 0,8,...120')
    return groups

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=pathlib.Path);p.add_argument('--checkpoint',required=True,type=pathlib.Path)
    p.add_argument('--dataset',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path)
    p.add_argument('--frames',type=int,choices=[16,32],default=16);p.add_argument('--device',default='auto');p.add_argument('--max-episodes',type=int)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    # Shared policy chooses only among externally visible devices.
    device_path=pathlib.Path(__file__).resolve().parents[1]/'core'/'e03'
    if device_path.exists():sys.path.insert(0,str(device_path))
    from device_utils import activate_device
    device=activate_device(a.device)
    device_info={'device':str(device),'cuda_visible_devices':__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}
    if device.type=='cuda':
        props=torch.cuda.get_device_properties(device)
        uuid=getattr(props,'uuid',None)
        if uuid is None:
            visible=device_info['cuda_visible_devices']
            if visible and visible.startswith(('GPU-','MIG-')):uuid=visible.split(',')[device.index]
        if uuid is None:
            rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()
            physical=str(device.index)
            if device_info['cuda_visible_devices']:physical=device_info['cuda_visible_devices'].split(',')[device.index].strip()
            uuid=dict(tuple(t.strip() for t in row.split(',',1)) for row in rows).get(physical)
        if uuid is None:raise RuntimeError('Cannot record selected GPU UUID')
        device_info.update(uuid=str(uuid),name=props.name,total_memory_bytes=props.total_memory)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_grad_enabled(False)
    model,info=load_encoder(a.source,a.checkpoint,device);transform=preprocessor()
    with np.load(a.dataset,allow_pickle=False) as f:data={k:f[k] for k in f.files}
    images=data['images']
    if images.dtype!=np.uint8 or images.ndim!=4 or images.shape[-1]!=3:raise ValueError('Require uint8 RGB NHWC')
    groups=episode_indices(data,a.frames)
    all_episode_count=len(groups)
    if a.max_episodes is not None:
        if a.max_episodes<=0:raise ValueError('max-episodes must be positive')
        groups=groups[:a.max_episodes]
    if not groups:raise ValueError('No episodes')
    means=[];grids=[];temporals=[];inputs=[];raws=[];ids_out=[];repeat_errors=[]
    with torch.inference_mode():
        for n,(episode,ids) in enumerate(groups):
            raw=images[ids];x=transform(raw)
            if tuple(x.shape)!=(3,a.frames,256,256) or x.dtype!=torch.float32:raise ValueError(f'Transform shape/dtype {x.shape}/{x.dtype}')
            if not torch.isfinite(x).all():raise ValueError('Nonfinite pixels')
            inputs.append(hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest());raws.append(hashlib.sha256(raw.tobytes()).hexdigest())
            x=x.unsqueeze(0).to(device);y=model(x)
            expected=(1,(a.frames//2)*256,1024)
            if tuple(y.shape)!=expected or not torch.isfinite(y).all():raise ValueError(f'Output invalid: {y.shape}')
            if n==0:
                again=model(x);repeat_errors.append(float((y-again).abs().max().cpu()))
                if repeat_errors[-1]>1e-6:raise AssertionError('FP32 repeat maxabs exceeds frozen 1e-6 tolerance')
            token=y[0].reshape(a.frames//2,16,16,1024)
            means.append(token.mean((0,1,2)).cpu().numpy())
            grids.append(token.reshape(a.frames//2,2,8,2,8,1024).mean((0,2,4)).reshape(4096).cpu().numpy())
            temporals.append(token.mean((1,2)).cpu().numpy());ids_out.append(ids)
            print(n+1,len(groups),episode,tuple(y.shape),flush=True)
    rows=np.array(ids_out);first=rows[:,0]
    np.savez_compressed(a.output/'features.npz',branch_vector=np.array(means,dtype=np.float32),grid=np.array(grids,dtype=np.float32),temporal_mean=np.array(temporals,dtype=np.float32))
    manifest={k:data[k][first] for k in ['labels','split','scene','contact_mode','mass_level','force_level','episode']}
    manifest.update(source_rows=rows,frame=data['frame'][first],episode_length=np.full(len(first),a.frames),clip_frames=data['frame'][rows],target_names=data['target_names'],raw_input_sha256=np.array(raws),processed_input_sha256=np.array(inputs))
    np.savez_compressed(a.output/'sample_manifest.npz',**manifest)
    # Equality is checked on actual complete clips and preprocessed pixels; no physics assertion is inherited.
    pairs=[]
    for scene in np.unique(manifest['scene']):
        for mode in ['on','off']:
            for pair,(l,r) in enumerate([((0,0),(1,1)),((1,0),(2,1))]):
                def find(level):
                    w=np.flatnonzero((manifest['scene']==scene)&(manifest['contact_mode']==mode)&(manifest['mass_level']==level[0])&(manifest['force_level']==level[1]))
                    return int(w[0]) if len(w)==1 else None
                i,j=find(l),find(r)
                if i is None or j is None:continue
                pairs.append({'scene':int(scene),'mode':mode,'pair':pair,'raw_equal':raws[i]==raws[j],
                              'processed_equal':inputs[i]==inputs[j],'mean_exact':bool(np.array_equal(means[i],means[j])),
                              'grid_exact':bool(np.array_equal(grids[i],grids[j])),
                              'mean_maxabs':float(np.max(np.abs(means[i]-means[j]))),'grid_maxabs':float(np.max(np.abs(grids[i]-grids[j])))})
    complete_audit=len(groups)==all_episode_count
    off_pairs=[p for p in pairs if p['mode']=='off']
    if complete_audit:
        expected_pairs=len(np.unique(manifest['scene']))*2
        if len(off_pairs)!=expected_pairs:raise AssertionError('Full audit missing OFF pairs')
        if not all(p['raw_equal'] and p['processed_equal'] and p['mean_maxabs']<=1e-6 and p['grid_maxabs']<=1e-6 for p in off_pairs):raise AssertionError('OFF same-input/equal-feature audit failed')
    if device.type=='cuda':
        torch.cuda.synchronize(device)
        device_info.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    info.update(device=str(device),device_metadata=device_info,torch_version=torch.__version__,cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
                frames=a.frames,frame_steps=[data['frame'][i].tolist() for _,i in groups],input_shape=[1,3,a.frames,256,256],
                output_token_shape=[1,(a.frames//2)*256,1024],output_layer='last encoder LayerNorm; no class token or predictor',
                pooling='branch_vector=all-token mean1024; grid=temporal mean and fixed2x2 spatial mean4096; temporal_mean=8x1024 for16frames',
                primary_feature_key='branch_vector',precision='FP32; no autocast; TF32 disabled',
                preprocessing='official Resize short_side292 bilinear; CenterCrop256; ClipToTensor /255; ImageNet mean/std; ndarray uint8 RGB',
                task_scope='whole-window retrospective parameter readout; never use tubelet timestamps as causal future prediction',
                dataset_sha256=sha(a.dataset),adapter_sha256=sha(__file__),repeat_maxabs=repeat_errors,equal_ratio_pairs=pairs)
    (a.output/'config.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    (a.output/'window_spec.json').write_text(json.dumps({'task':'retrospective_readout','sample_steps':data['frame'][groups[0][1]].tolist(),'pooling':'all-token mean1024','encoder_id':'vjepa2_vitl_target_encoder@'+COMMIT,'primary_feature_key':'branch_vector','source_dataset_sha256':info['dataset_sha256']},indent=2),encoding='utf-8')
    (a.output/'COMPLETE.json').write_text(json.dumps({'status':'PASS_FULL_CLIP_AUDIT' if complete_audit else 'PASS_SMOKE_ONLY','episode_count':len(groups),'source_episode_count':all_episode_count,'equality_checked':complete_audit,'equality_pairs_recorded':len(pairs),'off_pairs_checked':len(off_pairs),'repeatcheck_maxabs':repeat_errors,'repeatcheck_tolerance':1e-6},indent=2),encoding='utf-8')
    (a.output/'manifest.json').write_text(json.dumps({f.name:sha(f) for f in a.output.iterdir() if f.is_file()},indent=2),encoding='utf-8')
if __name__=='__main__':main()
