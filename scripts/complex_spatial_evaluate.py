"""Frozen spatial pooling holdout: train-only maps, tune-only ridge, sequence CIs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ALPHAS = (0., 1e-7, 1e-5, 1e-3, .01, .1, 1., 10., 100.)
SEED = 20261012


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''): h.update(b)
    return h.hexdigest()


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def ridge(xs, ys, device):
    """One thin SVD for all alphas/targets, including rank-safe alpha=0."""
    mx, my = xs[0].mean(0), ys[0].mean(0)
    x = [torch.as_tensor(v-mx, dtype=torch.float64, device=device) for v in xs]
    y = torch.as_tensor(ys[0]-my, dtype=torch.float64, device=device)
    u, s, vh = torch.linalg.svd(x[0], full_matrices=False)
    rhs = u.T @ y
    losses, weights = [], []
    cutoff = max(x[0].shape) * torch.finfo(s.dtype).eps * s[0]
    for a in ALPHAS:
        gain = torch.where(s > cutoff, 1/s.clamp_min(1e-300), 0.) if a == 0 else s/(s*s+len(x[0])*a)
        w = vh.T @ (gain[:, None]*rhs)
        weights.append(w)
        p = (x[1]@w).cpu().numpy()+my
        losses.append(np.mean((p-ys[1])**2, axis=0))
    losses = np.asarray(losses)
    chosen = losses.argmin(0)
    w = torch.stack(weights)[torch.as_tensor(chosen, device=device), :, torch.arange(len(my), device=device)].T
    pred = (x[2]@w).cpu().numpy()+my
    return pred, {'alpha': [ALPHAS[i] for i in chosen], 'tune_losses_by_alpha': losses.tolist(),
                  'train_feature_mean': mx.tolist(), 'train_target_mean': my.tolist(),
                  'weights': w.cpu().numpy()}


def legal_pairs(ep, fr):
    lookup = {(str(e), int(t)): i for i, (e,t) in enumerate(zip(ep,fr))}
    if len(lookup) != len(ep): raise ValueError('Duplicate episode/frame')
    pairs = [(i,lookup[(str(e),int(t)+16)]) for i,(e,t) in enumerate(zip(ep,fr)) if (str(e),int(t)+16) in lookup]
    return np.asarray(pairs, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    for name in ['root','input','output']: ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()): raise ValueError('Use a fresh output directory; no silent holdout reruns')
    sys.path.insert(0,str(args.root/'scripts')); sys.path.insert(0,str(args.root/'core/e03'))
    from device_utils import activate_device
    from round2_a_global import fit, invariants
    from e03_core import optimize_joint
    torch.set_num_threads(4)
    device=activate_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(SEED)
    start=time.time()
    with np.load(args.input/'sample_manifest.npz', allow_pickle=False) as z: m={k:z[k] for k in z.files}
    with np.load(args.input/'features.npz', allow_pickle=False) as z: features={k:z[k] for k in ['mean','grid']}
    masks=[m['split']==i for i in range(3)]
    assert [int(x.sum()) for x in masks]==[640,192,1280]
    episodes=[set(m['episode'][x].tolist()) for x in masks]
    assert [len(s) for s in episodes]==[10,3,20]
    assert all(not episodes[i]&episodes[j] for i in range(3) for j in range(i))
    assert all(np.sum(m['episode']==ep)==64 for ep in np.unique(m['episode']))
    assert features['mean'].shape==(2112,1280) and features['grid'].shape==(2112,5120)
    assert all(np.isfinite(x).all() for x in features.values())
    assert np.isfinite(m['labels']).all()
    names=m['target_names'].tolist()
    Y=[m['labels'][s].astype(np.float64) for s in masks]
    assert np.all(Y[0].std(0)>1e-8)
    pairs=legal_pairs(m['episode'][masks[0]],m['frame'][masks[0]])
    assert len(pairs)>128
    # The extractor flattened [N,1280,2,2]: channel-major, row-major cell order.
    grid=features['grid'].reshape(-1,1280,4)
    np.testing.assert_allclose(grid.mean(2),features['mean'],rtol=1e-5,atol=1e-5)
    perm=np.stack([np.random.default_rng(SEED+i).permutation(4) for i in range(len(grid))])
    features['shuffled']=np.take_along_axis(grid,perm[:,None,:],axis=2).reshape(-1,5120)
    # Keep the original labels, but de-duplicate the correlated background in headlines.
    bg=[i for i,n in enumerate(names) if n.startswith(('floor_color[','stage_color['))]
    bgmeta={}
    if bg:
        mean=Y[0][:,bg].mean(0); scale=Y[0][:,bg].std(0)
        _,bs,bv=np.linalg.svd((Y[0][:,bg]-mean)/scale,full_matrices=False)
        loading=bv[0]; loading*=1 if loading[np.argmax(abs(loading))]>=0 else -1
        Y=[np.column_stack([y,((y[:,bg]-mean)/scale)@loading]) for y in Y]
        names.append('background_train_rank1')
        bgmeta={'source_columns':bg,'mean':mean.tolist(),'std':scale.tolist(),'loading':loading.tolist(),'variance_fraction':float(bs[0]**2/(bs@bs))}
    std=Y[0].std(0)
    config={'scope':'Single frozen new-sequence holdout; same environment, static readout only',
            'primary':{'comparison':'mean128_white_full minus grid128_white_full',
                       'families':['object_xy','object_green'],'endpoint':'mean squared error normalized by train target std; equal weight per sequence',
                       'multiplicity':'Bonferroni familywise 95% across two fixed families (97.5% per-family intervals)'},
            'secondary':'rank64, spatial shuffle, other labels, sparse budgets and basis explanations; no winner selection on holdout',
            'ranks':[64,128],'main_rank':128,'alphas':ALPHAS,'budgets':[1,4,8,'full'],
            'nonorthogonal':{'seeds':[0,1,2],'steps':400,'active':2,'lam':.03},
            'block_steps':150,'pair_lag':16,'legal_pairs':len(pairs),
            'shuffle_seed_formula':'independent default_rng(20261012+sample_index).permutation(4)',
            'grid_layout':'channel-major N,1280,4; four cells row-major TL,TR,BL,BR; each frame shares one permutation across channels',
            'bootstrap':{'seed':20261013,'replicates':10000,'unit':'20 heldout sequences; paired; conditional on fitted train/tune models'},
            'background_train_only_rank1':bgmeta,'device':str(device),'torch':torch.__version__,
            'inputs':{n:sha(args.input/n) for n in ['features.npz','sample_manifest.npz']},
            'source_sha256':{str(p):sha(p) for p in [Path(__file__),args.root/'scripts/round2_a_global.py',args.root/'core/e03/e03_core.py']}}
    # This file is committed to disk before any representation fit or test score.
    write(args.output/'config.json',config)
    np.savez_compressed(args.output/'audit_arrays.npz',shuffle_per_sample=perm,legal_train_pairs=pairs,
                        target_train=Y[0],target_tune=Y[1],target_test=Y[2],target_names=names,
                        episode_test=m['episode'][masks[2]],frame_test=m['frame'][masks[2]],train_std=std)
    rows=[]; predictions={}; transforms={}; fits=[]; sequence_errors={}
    testep=m['episode'][masks[2]]; seq=np.unique(testep)

    def evaluate(key,Z,parents=None,full_only=False):
        cross=((Z[0]-Z[0].mean(0))/np.maximum(Z[0].std(0),1e-8)).T@((Y[0]-Y[0].mean(0))/std)/len(Z[0])
        for budget in ([1,4,8,Z[0].shape[1]] if Z[0].shape[1]>8 and not full_only else [Z[0].shape[1]]):
            tag='full' if budget==Z[0].shape[1] else f'top{budget}'
            out=np.empty_like(Y[2]); meta=[]
            # Full fit shares one SVD across all outputs; sparse fits use target-specific coordinates.
            if tag=='full':
                out,mm=ridge(Z,Y,device); transforms[key+'_'+tag+'_weights']=mm.pop('weights');meta=mm
            else:
                for j in range(len(names)):
                    ids=np.argsort(cross[:,j]**2,kind='stable')[-budget:]
                    pp,mm=ridge([z[:,ids] for z in Z],[y[:,j:j+1] for y in Y],device)
                    out[:,j]=pp[:,0];transforms[f'{key}_{tag}_target{j}_weights']=mm.pop('weights')
                    meta.append({'target':names[j],'coordinates':ids.tolist(),**mm})
            save(key+'_'+tag,out,meta)
        if parents is not None:
            for budget in [1,4]:
                out=np.empty_like(Y[2]); meta=[]
                for j,parent in enumerate(parents):
                    candidates=np.arange(parent*4,parent*4+4)
                    ids=candidates[np.argsort(cross[candidates,j]**2,kind='stable')[-budget:]]
                    pp,mm=ridge([z[:,ids] for z in Z],[y[:,j:j+1] for y in Y],device)
                    out[:,j]=pp[:,0];transforms[f'{key}_parent_top{budget}_target{j}_weights']=mm.pop('weights')
                    meta.append({'target':names[j],'parent':int(parent),'coordinates':ids.tolist(),**mm})
                save(key+f'_parent_top{budget}',out,meta)

    def save(key,p,meta):
        err=((p-Y[2])/std)**2
        per=np.asarray([err[testep==s].mean(0) for s in seq])
        denom=((Y[2]-Y[2].mean(0))**2).sum(0)
        r2=np.where(denom>1e-15,1-((p-Y[2])**2).sum(0)/np.maximum(denom,1e-15),np.nan)
        predictions[key]=p;sequence_errors[key]=per
        rows.append({'key':key,'r2':[float(v) if np.isfinite(v) else None for v in r2],
                     'normalized_mse':per.mean(0).tolist(),'per_sequence_normalized_mse':per.tolist(),'selection':meta})
        print('probe',key,flush=True)

    for pooling,raw in features.items():
        evaluate(f'{pooling}_raw',[raw[s].astype(np.float64) for s in masks],full_only=True)
        X=[torch.as_tensor(raw[s],device=device,dtype=torch.float64) for s in masks]
        mean=X[0].mean(0);center=X[0]-mean
        _,s,v=torch.linalg.svd(center,full_matrices=False)
        assert s[127]>s[0]*1e-8
        transforms[pooling+'_mean']=mean.cpu().numpy()
        for rank in [64,128]:
            basis=v[:rank].T/(s[:rank]/np.sqrt(len(center)-1))
            W=[((x-mean)@basis).cpu().numpy() for x in X]
            transforms[f'{pooling}{rank}_pca']=basis.cpu().numpy()
            fits.append({'key':f'{pooling}{rank}','variance_fraction':float(s[:rank].square().sum()/s.square().sum()),'whitening_error':float(np.max(abs(np.cov(W[0],rowvar=False)-np.eye(rank))))})
            evaluate(f'{pooling}{rank}_white',W)
            if rank!=128 or pooling=='shuffled': continue
            delta=W[0][pairs[:,1]]-W[0][pairs[:,0]]
            eig,sfa=np.linalg.eigh(delta.T@delta/len(delta))
            lag=W[0][pairs[:,0]].T@W[0][pairs[:,1]]/len(pairs)
            random=np.linalg.qr(np.random.default_rng(SEED).normal(size=(rank,rank)))[0]
            block,bfit=optimize_joint(lag[None],4,[np.eye(rank),sfa,random],steps=150,device=str(device))
            B=[w@block for w in W]
            co=(B[0]-B[0].mean(0)).T@((Y[0]-Y[0].mean(0))/std)/len(B[0])
            parents=(co.reshape(-1,4,len(names))**2).sum(1).argmax(0)
            transforms[pooling+'_sfa']=sfa;transforms[pooling+'_block4']=block;transforms[pooling+'_parents']=parents
            fits.append({'key':pooling+'_block4','fit':bfit,'sfa_eigenvalues':eig.tolist(),'parent_rule':'train explained energy in orthogonal block basis; shared with nonorthogonal seeds'})
            evaluate(f'{pooling}128_sfa',[w@sfa for w in W])
            evaluate(f'{pooling}128_full_block4',B,parents)
            for seed in [0,1,2]:
                D,history=fit(B[0].reshape(len(B[0]),-1,4),'nonorthogonal',seed,SimpleNamespace(steps=400,active=2,lam=.03),device)
                assert np.max(np.linalg.cond(D))<2.251
                checks=[invariants([b[:,i:i+4] for b in B],D[i//4]) for i in range(0,rank,4)]
                key=f'{pooling}128_nonorthogonal_seed{seed}'
                transforms[key+'_dictionary']=D
                fits.append({'key':key,'history':history,'invariants':checks,'max_condition':float(np.linalg.cond(D).max())})
                Z=[np.einsum('nbd,bdk->nbk',b.reshape(len(b),-1,4),np.linalg.inv(D)).reshape(len(b),-1) for b in B]
                evaluate(key,Z,parents)
        del X,center,s,v
    progress=(m['frame']/np.maximum(m['episode_length']-1,1))[:,None]
    evaluate('time_progress_oracle',[progress[s] for s in masks])
    evaluate('time_progress_cubic_oracle',[np.concatenate([progress[s]**i for i in [1,2,3]],1) for s in masks])
    families={'object_xy':['tool_block/cartesian_position[0]','tool_block/cartesian_position[1]'],
              'object_green':['tool_block/color[1]']}
    draws=np.random.default_rng(20261013).integers(0,len(seq),size=(10000,len(seq)))
    contrasts=[]
    for rank in [128,64]:
        delta=sequence_errors[f'mean{rank}_white_full']-sequence_errors[f'grid{rank}_white_full']
        for family,targets in families.items():
            ids=[names.index(n) for n in targets]
            values=delta[:,ids].mean(1); boot=values[draws].mean(1)
            contrasts.append({'rank':rank,'family':family,'targets':targets,'primary':rank==128,
                              'improvement_mean_minus_grid':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
                              'bonferroni_two_family_ci':np.quantile(boot,[.0125,.9875]).tolist(),
                              'per_sequence_difference':values.tolist(),'interpretation':'positive favors grid; no post-hoc method or endpoint selection'})
    np.savez_compressed(args.output/'predictions.npz',**predictions)
    np.savez_compressed(args.output/'transforms_and_weights.npz',**transforms)
    write(args.output/'results.json',{'targets':names,'sequence_ids':seq.tolist(),'rows':rows,'fits':fits,'primary_and_rank_sensitivity':contrasts,
          'headline_targets':[n for n in names if n.startswith(('tool_block/cartesian_position[','tool_block/color['))],
          'auxiliary':'joint positions, single-frame velocity, raw correlated background channels, sparse budgets and learned transformations',
          'limitations':['No new environment or physical causal intervention','Bootstrap conditions on fixed training and tune sequences','Joint scalar positions are auxiliary, not circular angle metrics','Nonorthogonal ordinary ridge changes coefficient penalty; invariance diagnostics separately use pulled-back penalty','No model winner selected from heldout scores']})
    write(args.output/'COMPLETE.json',{'complete':True,'seconds':time.time()-start,'conditions':len(rows),'source_sha256':sha(Path(__file__))})


if __name__=='__main__': main()

