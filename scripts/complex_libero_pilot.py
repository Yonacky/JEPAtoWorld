"""New-domain static-readout pilot. Demo-disjoint development data, never confirmation."""
import argparse, hashlib, json, pathlib, sys, time, types
import numpy as np

SEED = 20261011

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def load_hdf5(path, stride):
    import h5py
    out = {k: [] for k in ['images','labels','split','episode','frame','episode_length']}
    with h5py.File(path, 'r') as f:
        names = sorted(f['data'], key=lambda k: int(k.rsplit('_',1)[-1]))
        assert len(names) >= 30
        selected = np.random.default_rng(SEED).permutation(names)[:30].tolist()
        for j, name in enumerate(selected):
            obs = f['data'][name]['obs']
            n = len(obs['agentview_rgb']); ids = np.arange(0,n,stride)
            images = obs['agentview_rgb'][ids]
            assert images.shape[1:] == (128,128,3) and images.dtype == np.uint8
            # LIBERO macros_image_convention=opengl: one predeclared vertical flip.
            out['images'].append(images[:,::-1,:,:].copy())
            labels = np.concatenate([obs['ee_pos'][ids],obs['gripper_states'][ids]],axis=1)
            assert labels.shape == (len(ids),5)
            out['labels'].append(labels)
            out['split'].append(np.full(len(ids),0 if j<20 else 1 if j<25 else 2))
            out['episode'].append(np.repeat(name,len(ids)))
            out['frame'].append(ids); out['episode_length'].append(np.full(len(ids),n))
    data = {k: np.concatenate(v) for k,v in out.items()}
    data['target_names'] = np.array(['ee_pos_x','ee_pos_y','ee_pos_z','gripper_joint_0','gripper_joint_1'])
    return data, {'selected_demos':selected,'unused_demo_count':len(names)-30,'camera':'agentview_rgb','image_orientation':'fixed vertical flip from LIBERO opengl','labels':'same obs timestep ee_pos and gripper_states; no actions or simulator states read'}

def extract(images, model_root, device, batch, out):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    sys.path.insert(0,str(model_root/'vendor/ijepa-main'))
    from src.models.vision_transformer import vit_huge
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ck = model_root/'models/ijepa_vith14_1k/model.safetensors'
    h = load_file(str(ck)); d = {}; used = set()
    def put(dst,src): d[dst] = h[src]; used.add(src)
    put('pos_embed','embeddings.position_embeddings')
    for sf in ['weight','bias']:
        put('patch_embed.proj.'+sf,'embeddings.patch_embeddings.projection.'+sf)
        put('norm.'+sf,'layernorm.'+sf)
    for i in range(32):
        a=f'encoder.layer.{i}.'; b=f'blocks.{i}.'
        for sf in ['weight','bias']:
            keys=[a+'attention.attention.'+x+'.'+sf for x in ['query','key','value']]
            d[b+'attn.qkv.'+sf]=torch.cat([h[k] for k in keys]); used.update(keys)
            for dst,src in [('attn.proj','attention.output.dense'),('norm1','layernorm_before'),('norm2','layernorm_after'),('mlp.fc1','intermediate.dense'),('mlp.fc2','output.dense')]:
                put(b+dst+'.'+sf,a+src+'.'+sf)
    assert used == set(h)
    model=vit_huge(patch_size=14,img_size=[224]).float().eval()
    model.load_state_dict(d,strict=True); del h,d
    model.requires_grad_(False)
    def sdpa(self,x):
        b,n,c=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=self.scale)
        return self.proj_drop(self.proj(y.transpose(1,2).reshape(b,n,c))),None
    att=model.blocks[0].attn; x=torch.randn(2,16,1280)
    with torch.no_grad(): err=float((att(x)[0]-sdpa(att,x)[0]).abs().max())
    assert err<1e-5,err
    model=model.to(device); x=torch.randn(1,3,224,224,device=device)
    with torch.inference_mode(): reference=model(x)
    for block in model.blocks: block.attn.forward=types.MethodType(sdpa,block.attn)
    with torch.inference_mode(): actual=model(x)
    fullerr=float((reference-actual).abs().max())
    assert torch.allclose(actual,reference,atol=2e-4,rtol=2e-4),fullerr
    del reference,actual,x
    features=np.empty((len(images),1280),np.float32)
    with torch.inference_mode():
        for i in range(0,len(images),batch):
            x=torch.from_numpy(images[i:i+batch].copy()).permute(0,3,1,2).to(device=device,dtype=torch.float32)/255
            x=(F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)-.5)/.5
            y=model(x); assert y.shape[1:]==(256,1280) and y.dtype==torch.float32
            features[i:i+len(y)]=y.mean(1).cpu().numpy()
            print('features',i+len(y),len(images),flush=True)
    assert np.isfinite(features).all()
    meta={'images_sha256':hashlib.sha256(images.tobytes()).hexdigest(),'checkpoint_sha256':sha(ck),'strict_load':True,'all_checkpoint_keys_used':True,'sdpa_attention_maxabs':err,'sdpa_full_model_maxabs':fullerr,'sdpa_full_atol_rtol':2e-4,'precision':'FP32, TF32 disabled','resize':'224 square bilinear antialias=True align_corners=False','normalization':'RGB (x/255 - .5)/.5','pooling':'mean 256 tokens; 1280 dimensions','device':str(device)}
    np.savez(out/'features.npz',mean=features)
    dump(out/'features.json',meta)
    del model
    if device.type=='cuda': torch.cuda.empty_cache()
    return features,meta

def validate(data):
    n=len(data['images']); y=data['labels']
    assert y.ndim==2 and len(y)==n and np.isfinite(y).all()
    assert data['images'].dtype==np.uint8 and data['images'].shape[-1]==3
    assert set(np.unique(data['split']))=={0,1,2}
    for k in ['split','episode','frame','episode_length']: assert len(data[k])==n
    assert len(data['target_names'])==y.shape[1]
    groups=[set(data['episode'][data['split']==s].tolist()) for s in range(3)]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert np.all(y[data['split']==0].std(0)>1e-8), 'constant training target'
    for ep in np.unique(data['episode']):
        ids=np.flatnonzero(data['episode']==ep)
        assert np.all(np.diff(data['frame'][ids])>0)
        assert len(np.unique(data['episode_length'][ids]))==1
    assert np.all(data['frame']>=0) and np.all(data['frame']<data['episode_length'])

def evaluate(features,data,args,device):
    from round2_a_global import fit, probe, r2, invariants
    sys.path.insert(0,str(args.root/'core/e03'))
    from e03_core import optimize_joint
    masks=[data['split']==s for s in range(3)]
    X=[features[m].astype(np.float64) for m in masks]
    Y=[data['labels'][m].astype(np.float64) for m in masks]
    mean=X[0].mean(0); xc=X[0]-mean
    _,s,vh=np.linalg.svd(xc,full_matrices=False)
    d=min(args.dim,len(s)); assert d>=32 and d%4==0 and s[d-1]>s[0]*1e-8
    basis=vh[:d].T/(s[:d]/np.sqrt(len(xc)-1))
    W=[(x-mean)@basis for x in X]
    # Pair exclusively within a training demo and the explicit actual frame lag.
    ep=data['episode'][masks[0]]; fr=data['frame'][masks[0]]
    pair=np.flatnonzero((ep[1:]==ep[:-1]) & ((fr[1:]-fr[:-1])==args.stride))
    assert len(pair)>d, ('insufficient lawful SFA pairs',len(pair))
    delta=W[0][pair+1]-W[0][pair]
    val,sfa=np.linalg.eigh(delta.T@delta/len(delta))
    random=np.linalg.qr(np.random.default_rng(SEED).normal(size=(d,d)))[0]
    lag=W[0][pair].T@W[0][pair+1]/len(pair)
    block_basis,block_fit=optimize_joint(lag[None],4,[np.eye(d),sfa,random],steps=150,device=str(device))
    B=[w@block_basis for w in W]
    D,hist=fit(B[0].reshape(len(B[0]),-1,4),'nonorthogonal',SEED,args,device)
    invariant_checks=[invariants([b[:,i:i+4] for b in B],D[i//4]) for i in range(0,d,4)]
    assert np.max(np.linalg.cond(D))<2.251
    matrices={'pca_mean':mean,'pca_whitening':basis,'random':random,'sfa':sfa,'sfa_eigenvalues':val,'nonorthogonal_dictionary':D,'full_block4':block_basis,'legal_train_pair_starts':pair}
    np.savez(args.output/'transforms.npz',**matrices)
    representations={'white':W,'random_orthogonal':[w@random for w in W],'sfa':[w@sfa for w in W],'full_block4':B, 'nonorthogonal':[np.einsum('nbd,bdk->nbk',w.reshape(len(w),-1,4),np.linalg.inv(D)).reshape(len(w),-1) for w in B]}
    progress=(data['frame']/np.maximum(data['episode_length']-1,1))[:,None]
    representations['time_progress_oracle']=[progress[m] for m in masks]
    # Polynomial progress controls are diagnostics; duration is privileged trajectory metadata.
    representations['time_progress_cubic_oracle']=[np.concatenate([progress[m]**p for p in [1,2,3]],axis=1) for m in masks]
    parents={}
    for j in range(Y[0].shape[1]):
        yy=Y[0][:,j:j+1]-Y[0][:,j:j+1].mean(0)
        energies=[]
        for b in range(d//4):
            xx=B[0][:,b*4:b*4+4]; xx=xx-xx.mean(0)
            fitted=xx@np.linalg.lstsq(xx,yy,rcond=None)[0]
            energies.append(float(np.sum(fitted**2)))
        parents[j]=int(np.argmax(energies))
    rows=[]; predictions={'target_dev':Y[2],'episode_dev':data['episode'][masks[2]],'frame_dev':data['frame'][masks[2]]}
    for method,Z in representations.items():
        budgets=[1,2,4,8,d] if Z[0].shape[1]==d else [Z[0].shape[1]]
        for j,name in enumerate(data['target_names'].tolist()):
            yy=[y[:,j:j+1] for y in Y]
            for budget in budgets:
                pred,meta=probe(Z,yy,budget)
                mse=np.mean((pred-yy[2])**2)
                per_demo={str(e):float(np.mean((pred[data['episode'][masks[2]]==e]-yy[2][data['episode'][masks[2]]==e])**2)) for e in np.unique(data['episode'][masks[2]])}
                key=f'{method}_{name}_k{budget}'; predictions[key]=pred
                rows.append({'method':method,'target':name,'budget':budget,'r2':r2(yy[2],pred),'rmse':float(np.sqrt(mse)),'mse_per_demo':per_demo,**meta})
    for method in ['full_block4','nonorthogonal']:
        Z=representations[method]
        for j,name in enumerate(data['target_names'].tolist()):
            parent=parents[j]; zz=[z[:,parent*4:parent*4+4] for z in Z]; yy=[y[:,j:j+1] for y in Y]
            for budget in [1,2,4]:
                pred,meta=probe(zz,yy,budget)
                predictions[f'{method}_{name}_parent{parent}_k{budget}']=pred
                rows.append({'method':method,'scope':'shared_parent_block','target':name,'parent':parent,'budget':budget,'r2':r2(yy[2],pred),'rmse':float(np.sqrt(np.mean((pred-yy[2])**2))),**meta})
    np.savez(args.output/'predictions.npz',**predictions)
    dump(args.output/'results.json',{'status':'new_domain_pilot_development_not_confirmation','rows':rows,'pca_dim':d,'pca_fit':'training features only, labels absent','sfa_pairs':len(pair),'sfa_actual_frame_lag':args.stride,'nonorthogonal_history':hist,'block4_fit':block_fit,'block_invariants':invariant_checks,'shared_parent_selection':'training OLS explained energy in full_block4; frozen for nonorthogonal comparison','max_block_condition':float(np.linalg.cond(D).max()),'readout':'per-scalar train correlation top-k selection; train ridge fit, tune alpha; dev scoring; no refit on tune','limitations':['Single task; not a final held-out confirmation.','Static pose readout only; no forecast or planning.','Top-k selection is supervised readout diagnosis, not unsupervised transform selection.','Nonorthogonal blocks have more degrees of freedom than orthogonal blocks.','Time progress uses true full demonstration duration and is an oracle diagnostic.','Frames within demos are correlated; per-demo errors retained.']})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=pathlib.Path,required=True)
    g=p.add_mutually_exclusive_group(required=True); g.add_argument('--hdf5',type=pathlib.Path); g.add_argument('--prepared',type=pathlib.Path)
    p.add_argument('--model-root',type=pathlib.Path,required=True); p.add_argument('--output',type=pathlib.Path,required=True)
    p.add_argument('--device',default='auto'); p.add_argument('--stride',type=int,default=4); p.add_argument('--dim',type=int,choices=[32,64],default=64)
    p.add_argument('--features-cache',type=pathlib.Path,help='Existing features.npz; sibling sample_manifest.npz must match exactly')
    p.add_argument('--batch-size',type=int,default=16); p.add_argument('--steps',type=int,default=400)
    p.add_argument('--active',type=int,default=2); p.add_argument('--lam',type=float,default=.03)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.root/'scripts'))
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0,str(args.root/'core/e03'))
    from device_utils import activate_device
    device=activate_device(args.device)
    start=time.time()
    if args.prepared:
        with np.load(args.prepared,allow_pickle=False) as z: data={k:z[k] for k in z.files}
        source={'prepared_sha256':sha(args.prepared),'orientation':'prepared images consumed unchanged'}
    else: data,source=load_hdf5(args.hdf5,args.stride)
    validate(data)
    np.savez(args.output/'sample_manifest.npz',**{k:v for k,v in data.items() if k!='images'})
    from PIL import Image
    Image.fromarray(data['images'][0]).save(args.output/'first_upright_frame.png')
    config={'seed':SEED,'source':source,'source_script_sha256':sha(pathlib.Path(__file__)),'args':{k:str(v) if isinstance(v,pathlib.Path) else v for k,v in vars(args).items()},'split_frame_counts':[int(np.sum(data['split']==s)) for s in range(3)],'split_demo_counts':[len(np.unique(data['episode'][data['split']==s])) for s in range(3)],'torch':torch.__version__,'numpy':np.__version__}
    dump(args.output/'config.json',config)
    if args.features_cache:
        with np.load(args.features_cache.parent/'sample_manifest.npz',allow_pickle=False) as old:
            for k in ['labels','split','episode','frame','episode_length','target_names']:
                assert np.array_equal(data[k],old[k]), ('feature cache sample mismatch',k)
        with np.load(args.features_cache,allow_pickle=False) as cached: features=cached['mean']
        assert features.shape==(len(data['images']),1280) and np.isfinite(features).all()
        meta=json.loads((args.features_cache.parent/'features.json').read_text())
        assert meta['precision']=='FP32, TF32 disabled' and meta['strict_load']
        assert meta['images_sha256']==hashlib.sha256(data['images'].tobytes()).hexdigest()
        assert meta['checkpoint_sha256']==sha(args.model_root/'models/ijepa_vith14_1k/model.safetensors')
        config['reused_features_sha256']=sha(args.features_cache)
        dump(args.output/'config.json',config)
    else:
        features,meta=extract(data['images'],args.model_root,device,args.batch_size,args.output)
    evaluate(features,data,args,device)
    dump(args.output/'COMPLETE.json',{'status':'new_domain_pilot_development_not_confirmation','seconds':time.time()-start,'source_script_sha256':config['source_script_sha256']})

if __name__=='__main__': main()
