"""Predeclared Haar common/detail readouts on sequence-disjoint heldout data."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
from complex_spatial_evaluate import ALPHAS, sha, write

SEED=20261016
H=.5*np.asarray([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=np.float64)
BUDGETS=((64,64),(32,96),(96,32),(0,128),(128,0))
RATIOS=(0.,.1,.3,1.,3.,10.)


def fit_readout(X,Y,device,common=None,ratios=(1.,)):
    """Fit all candidates on train, select only by tune, then predict test.

    Returned coefficients multiply centered, unscaled input columns. Thus every
    candidate, including ratio=0 (C-only), is reconstructible from saved arrays.
    """
    mx=X[0].mean(0); my=Y[0].mean(0)
    x=[torch.as_tensor(v-mx,dtype=torch.float64,device=device) for v in X]
    yt=torch.as_tensor(Y[0]-my,dtype=torch.float64,device=device)
    losses=[]; all_w=[]; candidates=[]
    for ratio in ratios:
        scale=torch.ones(X[0].shape[1],dtype=torch.float64,device=device)
        if common is not None: scale[common:]=ratio
        u,s,v=torch.linalg.svd(x[0]*scale,full_matrices=False)
        rhs=u.T@yt; threshold=max(x[0].shape)*torch.finfo(s.dtype).eps*s[0]
        for alpha in ALPHAS:
            gain=torch.where(s>threshold,1/s.clamp_min(1e-300),0.) if alpha==0 else s/(s*s+len(x[0])*alpha)
            w=scale[:,None]*(v.T@(gain[:,None]*rhs))
            losses.append(np.mean(((x[1]@w).cpu().numpy()+my-Y[1])**2,axis=0))
            all_w.append(w.cpu().numpy()); candidates.append([ratio,alpha])
    losses=np.asarray(losses); all_w=np.asarray(all_w)
    best=losses.argmin(0)  # fixed ratio-major/alpha-minor tie ordering
    weights=all_w[best,:,np.arange(Y[0].shape[1])].T
    pred=(X[2]-mx)@weights+my
    return pred,{'mean_x':mx,'mean_y':my,'all_weights':all_w,'selected_weights':weights,
                 'tune_mse':losses,'selected_candidate':best,'ratio_alpha_candidates':np.asarray(candidates)}


def main():
    p=argparse.ArgumentParser()
    for k in ['root','input','output']: p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='auto');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()): raise ValueError('Output must be fresh; do not overwrite a scored run')
    sys.path.insert(0,str(a.root/'core/e03'))
    from device_utils import activate_device
    device=activate_device(a.device);torch.set_num_threads(4)
    torch.manual_seed(SEED);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.time()
    with np.load(a.input/'sample_manifest.npz',allow_pickle=False) as z: m={k:z[k] for k in z.files}
    with np.load(a.input/'features.npz',allow_pickle=False) as z: f={k:z[k].astype(np.float64) for k in ['mean','grid']}
    masks=[m['split']==i for i in range(3)]
    assert [int(q.sum()) for q in masks]==[640,192,1280]
    episodes=[set(m['episode'][q].tolist()) for q in masks]
    assert [len(q) for q in episodes]==[10,3,20]
    assert episodes[2]=={30,155,114,182,89,10,106,165,171,13,87,88,105,92,111,167,17,138,3,196}, 'Unexpected heldout sequence IDs'
    assert all(not episodes[i]&episodes[j] for i in range(3) for j in range(i))
    assert all(np.sum(m['episode']==e)==64 for e in np.unique(m['episode']))
    assert f['mean'].shape==(2112,1280) and f['grid'].shape==(2112,5120)
    assert all(np.isfinite(q).all() for q in f.values()) and np.isfinite(m['labels']).all()
    Y=[m['labels'][q].astype(np.float64) for q in masks];names=m['target_names'].tolist()
    std=Y[0].std(0);assert np.all(std>1e-8)
    grid=f['grid'].reshape(-1,1280,4);haar=grid@H.T
    np.testing.assert_allclose(H@H.T,np.eye(4),atol=1e-14)
    np.testing.assert_allclose(haar[:,:,0],2*f['mean'],rtol=1e-5,atol=2e-5)
    f['C']=haar[:,:,0];f['D']=haar[:,:,1:].reshape(-1,3840)
    families={'object_xy':['tool_block/cartesian_position[0]','tool_block/cartesian_position[1]'],
              'object_green':['tool_block/color[1]']}
    for family in families.values(): assert all(n in names for n in family)
    config={'scope':'Predeclared new-sequence confirmation for fixed C64+D64 vs grid128 on newly locked 20-sequence holdout; same environment; secondary readouts exploratory',
            'primary_predeclared_for_this_run':'C64_D64_full ordinary ridge vs grid128_full; no budget selection on test',
            'endpoint':'sequence-equal train-standardized MSE improvement grid minus Haar; positive favors Haar',
            'families':families,'CI':'paired 20-sequence percentile bootstrap; 10000 draws; 97.5% per family, Bonferroni two-family 95%',
            'seed':SEED,'H':H.tolist(),'layout':'grid N,1280,4 channel-major row-major cells TL TR BL BR; h=grid@H.T; C=h[:,:,0]; D=h[:,:,1:].reshape(N,3840)',
            'budgets':BUDGETS,'ratios':RATIOS,'alphas':ALPHAS,'secondary':'all other C/D budgets, group scales, joint whitening, raw-feature invariance, top1/4/8 readouts',
            'fitting':'train-only separate C/D PCA whitened with n-1; label correlation top-k on train; per-target alpha and ratio selected on tune only; these labeled readout choices are supervised model selection, not a purely unsupervised discovery',
            'group_penalty':'C unchanged, D multiplied by ratio; ratio0 suppresses D; isotropic ridge in scaled coordinates; save weights pulled back to original input coordinates',
            'tie_order':'ratio-major ascending declared order, alpha-minor ALPHAS order; np.argmin first',
            'device':str(device),'torch':torch.__version__,'numpy':np.__version__,
            'inputs':{k:sha(a.input/k) for k in ['features.npz','sample_manifest.npz']},
            'source_sha256':sha(Path(__file__)),'helper_sha256':sha(Path(__file__).with_name('complex_spatial_evaluate.py'))}
    write(a.output/'config.json',config) # before fitting or calculating any test performance
    testep=m['episode'][masks[2]];seq=np.unique(testep)
    np.savez_compressed(a.output/'audit_arrays.npz',target_train=Y[0],target_tune=Y[1],target_test=Y[2],train_std=std,
                        target_names=names,episode_test=testep,frame_test=m['frame'][masks[2]],H=H)
    maps={}; P={}; diagnostics=[]
    for kind in ['mean','grid','C','D']:
        raw=torch.as_tensor(f[kind],device=device,dtype=torch.float64)
        train=raw[torch.as_tensor(masks[0],device=device)]
        mean=train.mean(0);_,s,v=torch.linalg.svd(train-mean,full_matrices=False)
        assert s[127]>s[0]*1e-8
        basis=v[:128].T/(s[:128]/np.sqrt(len(train)-1))
        P[kind]=[((raw[torch.as_tensor(q,device=device)]-mean)@basis).cpu().numpy() for q in masks]
        maps[kind+'_mean']=mean.cpu().numpy(); maps[kind+'_basis128']=basis.cpu().numpy();maps[kind+'_singular_values']=s.cpu().numpy()
        diagnostics.append({'kind':kind,'variance_fraction128':float(s[:128].square().sum()/s.square().sum()),
                            'whitening_error':float(abs(np.cov(P[kind][0],rowvar=False)-np.eye(128)).max())})
        print('PCA',kind,flush=True)
    np.savez_compressed(a.output/'transforms.npz',**maps)
    rows=[];predictions={};errors={};selections={}
    def record(key,X,common=None,ratios=(1.,),targets=Y,coordinates=None):
        pred,fit=fit_readout(X,targets,device,common,ratios)
        if coordinates is not None: fit['coordinates']=coordinates
        np.savez_compressed(a.output/(key+'_fit.npz'),**fit)
        return pred,fit
    def score(key,pred,selection):
        predictions[key]=pred;selections[key]=selection
        err=((pred-Y[2])/std)**2
        per=np.asarray([err[testep==s].mean(0) for s in seq]);errors[key]=per
        den=((Y[2]-Y[2].mean(0))**2).sum(0)
        rr=1-((pred-Y[2])**2).sum(0)/np.maximum(den,1e-15)
        rows.append({'key':key,'nmse':per.mean(0).tolist(),'per_sequence_nmse':per.tolist(),
                     'r2':[float(v) if d>1e-15 else None for v,d in zip(rr,den)],'selection':selection})
        print('readout',key,flush=True)
    representations={'mean128':P['mean'],'grid128':P['grid']}
    for c,d in BUDGETS:
        representations[f'C{c}_D{d}']=[np.concatenate([pc[:,:c],pd[:,:d]],1) for pc,pd in zip(P['C'],P['D'])]
    # Predeclared secondary control: remove residual cross-group covariance.
    joint=representations['C64_D64'];jm=joint[0].mean(0)
    jt=torch.as_tensor(joint[0]-jm,dtype=torch.float64,device=device)
    _,js,jv=torch.linalg.svd(jt,full_matrices=False)
    keep=js>js[0]*1e-10  # numerical nulls only, no target-driven rank selection
    jb=(jv[keep].T/(js[keep]/np.sqrt(len(jt)-1))).cpu().numpy()
    representations['C64_D64_joint_white']=[(q-jm)@jb for q in joint]
    cross=(joint[0][:,:64]-jm[:64]).T@(joint[0][:,64:]-jm[64:])/(len(jt)-1)
    diagnostics.append({'kind':'C64_D64_joint_white','rank':int(keep.sum()),
                        'numerical_null_cutoff_relative':1e-10,
                        'cross_group_covariance_frobenius':float(np.linalg.norm(cross)),
                        'cross_group_canonical_singular_values':np.linalg.svd(cross,compute_uv=False).tolist()})
    np.savez_compressed(a.output/'joint_whitening.npz',mean=jm,basis=jb,singular_values=js.cpu().numpy(),cross_group_covariance=cross)
    for key,X in representations.items():
        pred,fit=record(key+'_full',X)
        score(key+'_full',pred,{'ratio_alpha':fit['ratio_alpha_candidates'][fit['selected_candidate']].tolist()})
        cross=((X[0]-X[0].mean(0))/np.maximum(X[0].std(0),1e-8)).T@((Y[0]-Y[0].mean(0))/std)/len(X[0])
        for k in [1,4,8]:
            pred=np.empty_like(Y[2]);select=[]
            for j,name in enumerate(names):
                ids=np.argsort(cross[:,j]**2,kind='stable')[-k:]
                pp,fit=record(f'{key}_top{k}_target{j}',[x[:,ids] for x in X],targets=[y[:,j:j+1] for y in Y],coordinates=ids)
                pred[:,j]=pp[:,0];select.append({'target':name,'coordinates':ids.tolist(),'ratio_alpha':fit['ratio_alpha_candidates'][fit['selected_candidate']].tolist()})
            score(f'{key}_top{k}',pred,select)
        if key.startswith('C') and key!='C64_D64_joint_white':
            c=int(key.split('_')[0][1:])
            pred,fit=record(key+'_group_scaled',X,common=c,ratios=RATIOS)
            score(key+'_group_scaled',pred,{'ratio_alpha':fit['ratio_alpha_candidates'][fit['selected_candidate']].tolist(),
                                           'role':'secondary tune-selected penalty; no test selection'})
    # Orthogonal full-dimensional Haar must preserve isotropic-ridge predictions.
    raw_grid=[f['grid'][q] for q in masks];raw_haar=[haar.reshape(-1,5120)[q] for q in masks]
    pg,fg=record('raw_grid',raw_grid);ph,fh=record('raw_haar',raw_haar)
    # Audit all candidates, not only a possibly tie-sensitive chosen alpha.
    np.testing.assert_allclose(fg['tune_mse'],fh['tune_mse'],rtol=1e-5,atol=1e-8)
    # Move Haar coefficients back into grid channel/cell coordinates.
    mapped=np.einsum('ab,rnbt->rnat',H.T,fh['all_weights'].reshape(len(ALPHAS),1280,4,len(names))).reshape(len(ALPHAS),5120,len(names))
    candidate_error=0.
    for i in range(len(ALPHAS)):
        ga=(raw_grid[2]-fg['mean_x'])@fg['all_weights'][i]+fg['mean_y']
        ha=(raw_haar[2]-fh['mean_x'])@fh['all_weights'][i]+fh['mean_y']
        candidate_error=max(candidate_error,float(np.max(abs(ga-ha))))
        np.testing.assert_allclose(ga,ha,rtol=1e-5,atol=1e-6)
    invariance={'all_candidate_test_maxabs':candidate_error,'selected_alphas_equal':bool(np.array_equal(fg['selected_candidate'],fh['selected_candidate'])),
                'selected_prediction_maxabs':float(abs(pg-ph).max()),'mapped_coefficient_maxabs':float(abs(mapped-fg['all_weights']).max()),
                'meaning':'same full raw feature span and isotropic coefficient norm; no PCA truncation'}
    np.testing.assert_allclose(pg,ph,rtol=1e-5,atol=1e-6)
    score('raw_grid',pg,{'audit':'raw isotropic ridge'});score('raw_haar',ph,{'audit':'orthogonal raw isotropic ridge'})
    delta=errors['grid128_full']-errors['C64_D64_full']
    draws=np.random.default_rng(SEED).integers(0,len(seq),size=(10000,len(seq)));comparisons=[]
    for family,targets in families.items():
        ids=[names.index(n) for n in targets];values=delta[:,ids].mean(1);boot=values[draws].mean(1)
        comparisons.append({'family':family,'targets':targets,'improvement_grid_minus_haar64_64':float(values.mean()),
                            'ci975_bonferroni':np.quantile(boot,[.0125,.9875]).tolist(),'ci95_descriptive':np.quantile(boot,[.025,.975]).tolist(),
                            'per_sequence_improvement':values.tolist()})
    np.savez_compressed(a.output/'predictions.npz',**predictions)
    write(a.output/'results.json',{'targets':names,'sequence_ids':seq.tolist(),'rows':rows,'pca':diagnostics,
                                  'primary_comparisons':comparisons,'raw_orthogonal_invariance':invariance,
                                  'limitations':['Exact newly locked 20 test sequence IDs asserted; train/tune/test disjoint; same environment rather than cross-environment confirmation',
                                                 'CI conditional on fixed train/tune and selected readouts',
                                                 'No physical causal intervention or disentanglement claim','All allocation and penalty comparisons secondary, no heldout winner selection']})
    write(a.output/'COMPLETE.json',{'complete':True,'seconds':time.time()-started,'conditions':len(rows),'source_sha256':sha(Path(__file__))})


if __name__=='__main__': main()



