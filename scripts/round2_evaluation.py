"""Shared exploratory diagnostics. Fit on train, select ridge on tune, score dev.
Nonorthogonal ridge equivalence uses the pulled-back original coefficient norm.
No factor labels are used to learn representation transformations here.
"""
import argparse
import json
import os
import pathlib
import hashlib
import numpy as np
import torch

ALPHAS = (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.)

def choose_device():
    if torch.cuda.is_available():
        available = []
        for i in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(i):
                    torch.empty(1, device=f'cuda:{i}')
                    available.append((torch.cuda.mem_get_info()[0], i))
            except RuntimeError:
                pass
        if available:
            return torch.device(f'cuda:{max(available)[1]}')
    return torch.device('cpu')

def ridge_predictions(xs, ys, alphas=ALPHAS, penalty=None):
    """Return dev predictions and train-only normalization; alpha per output."""
    mx, my = xs[0].mean(0), ys[0].mean(0)
    x = [v-mx for v in xs]
    y = [v-my for v in ys]
    gram = x[0].T @ x[0] / len(x[0])
    rhs = x[0].T @ y[0] / len(x[0])
    if penalty is None:
        penalty = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    predictions, tune = [], []
    for alpha in alphas:
        w = torch.linalg.solve(gram + alpha*penalty, rhs)
        predictions.append(x[2] @ w + my)
        tune.append(((x[1] @ w - y[1])**2).mean(0))
    losses = torch.stack(tune)
    selected = losses.argmin(0)
    stack = torch.stack(predictions)
    pred = stack[selected, :, torch.arange(ys[0].shape[1], device=stack.device)].T
    return pred, [float(alphas[j]) for j in selected.cpu().tolist()]

def r2(y, p):
    denom = ((y-y.mean(0))**2).sum(0)
    out = 1 - ((y-p)**2).sum(0) / denom.clamp_min(1e-15)
    out[denom <= 1e-15] = float('nan')
    return out

def paired_block_mse_interval(y, a, b, train_std, indices, block_size=64, repeats=500, seed=0):
    """Exploratory conditional uncertainty, not independent trajectory confirmation.
    Difference is MSE(a)-MSE(b), standardized using training target scale.
    Resamples complete contiguous nonoverlapping blocks; drops incomplete tail.
    """
    if len(y) != len(indices) or not np.all(np.diff(indices)==1):
        raise ValueError('Requires audited contiguous source sequence')
    nblocks = len(y)//block_size
    if nblocks < 8:
        raise ValueError('Insufficient temporal blocks')
    scale = np.maximum(np.asarray(train_std), 1e-8)
    error = ((y-a)/scale)**2 - ((y-b)/scale)**2
    means = error[:nblocks*block_size].reshape(nblocks, block_size, -1).mean(1)
    draws = np.random.default_rng(seed).integers(0, nblocks, size=(repeats,nblocks))
    boot = means[draws].mean(1)
    return {'mean_difference':means.mean(0).tolist(), 'interval_025_975':np.quantile(boot,[.025,.975],axis=0).tolist(),
            'block_size':block_size,'blocks':nblocks,'discarded_tail':len(y)-nblocks*block_size,
            'scope':'conditional on fitted model and reused exploratory sequence; no trajectory IDs available'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=pathlib.Path(args.root); out=pathlib.Path(args.output); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    device=choose_device()
    rows=[]; checks=[]; intervals=[]; hashes={}
    for ds in ('pong','pinball'):
        src=root/'runs/e04_inputs/cache'/f'{ds}_ijepa_r0.npz'
        hashes[str(src.relative_to(root))]=hashlib.sha256(src.read_bytes()).hexdigest()
        cache=np.load(src)
        schema=json.loads((root/'experiments/e03_fp32_20261011'/f'{ds}_factor_schema.json').read_text())
        groups=[g for g in schema if g['type']=='continuous']
        columns=[g['slice'][0] for g in groups]
        names=[g['name'] for g in groups]
        xs=[torch.as_tensor(cache[s],dtype=torch.float64,device=device) for s in ('train','tune','dev')]
        ys=[torch.as_tensor(cache['target_'+s][:,columns],dtype=torch.float64,device=device) for s in ('train','tune','dev')]
        original,alpha=ridge_predictions(xs,ys)
        # Known invertible nonorthogonal transform, independent of labels/data.
        rng=np.random.default_rng(67)
        qa=np.linalg.qr(rng.normal(size=(128,128)))[0]
        qb=np.linalg.qr(rng.normal(size=(128,128)))[0]
        mat=torch.as_tensor(qa@np.diag(np.geomspace(.5,2,128))@qb.T,dtype=torch.float64,device=device)
        transformed=[x@mat for x in xs]
        pulled,ap=ridge_predictions(transformed,ys,penalty=mat.T@mat)
        ordinary,ao=ridge_predictions(transformed,ys)
        diff=float((original-pulled).abs().max())
        assert diff<1e-8 and alpha==ap, (ds,diff)
        checks.append({'dataset':ds,'transform_condition_number':4.,'pulled_back_ridge_max_prediction_error':diff,
                       'pulled_back_same_selected_alphas':alpha==ap,'ordinary_reoptimized_ridge_prediction_difference':float((ordinary-original).abs().max())})
        prediction_store={'target':ys[2].cpu().numpy(),'dense_original':original.cpu().numpy(),'dense_pulled_back':pulled.cpu().numpy()}
        scores={}
        for method in ('white','sfa','full_block4'):
            paths=list((root/'runs/e04_inputs/transforms').rglob(f'{ds}_ijepa_r0_{method}.npz'))
            if len(paths)!=1: raise ValueError(paths)
            basis=np.load(paths[0])['past_basis']
            hashes[str(paths[0].relative_to(root))]=hashlib.sha256(paths[0].read_bytes()).hexdigest()
            q=torch.as_tensor(basis,dtype=torch.float64,device=device)
            xx=[x@q for x in xs]
            centered=xx[0]-xx[0].mean(0)
            std=ys[0].std(0,correction=0).clamp_min(1e-8)
            cross=centered.T@((ys[0]-ys[0].mean(0))/std)/len(centered)
            cov=centered.T@centered/len(centered)
            white_error=float((cov-torch.eye(128,device=device,dtype=torch.float64)).abs().max())
            assert white_error<1e-6, white_error
            # Orthogonal whitened groups only: train cross-energy is valid selector.
            blockscore=(cross.reshape(32,4,-1)**2).sum(1)
            selected=blockscore.argmax(0)
            pred=torch.empty_like(ys[2]); selected_alphas=[]
            for j in range(len(names)):
                bi=int(selected[j]); sl=slice(bi*4,bi*4+4)
                pp, aa=ridge_predictions([x[:,sl] for x in xx],[y[:,j:j+1] for y in ys])
                pred[:,j]=pp[:,0];selected_alphas.append(aa[0])
            scores[method]=pred.cpu().numpy()
            prediction_store[method+'_block4']=scores[method]
            rows.append({'dataset':ds,'method':method,'factors':names,'block4_r2':r2(ys[2],pred).cpu().tolist(),
                         'dense_r2':r2(ys[2],original).cpu().tolist(),'blocks':selected.cpu().tolist(),
                         'readout_alphas':selected_alphas,'whitening_error':white_error,
                         'selection':'factor-supervised train diagnostic blocks; representation transforms unsupervised; alpha on tune'})
            print(ds,method,'block4 mean R2',float(r2(ys[2],pred).mean()),flush=True)
        for bs in (32,64,128):
            for baseline in ('white','sfa'):
                item=paired_block_mse_interval(ys[2].cpu().numpy(),scores[baseline],scores['full_block4'],
                    ys[0].std(0,correction=0).cpu().numpy(),cache['source_dev'],bs)
                item.update(dataset=ds,factors=names,contrast=f'{baseline}-full_block4')
                intervals.append(item)
        np.savez_compressed(out/f'{ds}_predictions.npz',**prediction_store)
    report={'status':'complete','scope':'reused r0 exploratory continuous-factor diagnostics, not confirmation',
            'device':str(device),'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'source_sha256':hashes,
            'baselines':rows,'invariance_checks':checks,'block_bootstrap':intervals,
            'limitations':['temporal blocks are not verified independent trajectories','block-size sensitivity shown; no multiple-comparison correction','factorwise ridge selection differs from categorical or angular grouped metrics']}
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (out/'COMPLETE.json').write_text(json.dumps({'complete':True,'checks_passed':True,'datasets':2,'conditions':6}))
    print('COMPLETE',out,flush=True)

if __name__=='__main__': main()
