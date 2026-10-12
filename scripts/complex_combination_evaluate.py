"""Exploratory parity-combination split of two previously inspected holdouts."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
from complex_haar_evaluate import H, fit_readout
from complex_spatial_evaluate import ALPHAS, sha, write

SEED=20261020
FAMILIES={'position_xy':['tool_block/cartesian_position[0]','tool_block/cartesian_position[1]'],
          'green':['tool_block/color[1]']}


def load(path):
    with np.load(path/'sample_manifest.npz',allow_pickle=False) as z: m={k:z[k] for k in z.files}
    with np.load(path/'features.npz',allow_pickle=False) as z: f={k:z[k] for k in ['mean','grid']}
    assert [int(sum(m['split']==i)) for i in range(3)]==[640,192,1280]
    assert [len(np.unique(m['episode'][m['split']==i])) for i in range(3)]==[10,3,20]
    assert f['mean'].shape==(2112,1280) and f['grid'].shape==(2112,5120)
    assert all(np.isfinite(v).all() for v in f.values()) and np.isfinite(m['labels']).all()
    return m,f


def interval(values,sequence_ids,seed):
    valid=np.isfinite(values); values=values[valid]; ids=sequence_ids[valid]
    if len(values)<2: return {'mean':None,'ci95':None,'sequences':ids.tolist(),'n_sequences':len(ids)}
    draws=np.random.default_rng(seed).integers(len(values),size=(10000,len(values)))
    return {'mean':float(values.mean()),'ci95':np.quantile(values[draws].mean(1),[.025,.975]).tolist(),
            'sequences':ids.tolist(),'n_sequences':len(ids),'per_sequence':values.tolist()}


def main():
    p=argparse.ArgumentParser()
    for k in ['root','input-a','input-b','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='auto');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Use fresh output directory')
    started=time.time();sys.path.insert(0,str(a.root/'core/e03'))
    from device_utils import activate_device
    device=activate_device(a.device);torch.set_num_threads(4);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ma,fa=load(a.input_a);mb,fb=load(a.input_b)
    assert np.array_equal(ma['target_names'],mb['target_names'])
    for split in [0,1]:
        ia=ma['split']==split;ib=mb['split']==split
        for k in ['labels','episode','frame','episode_length']:
            assert np.array_equal(ma[k][ia],mb[k][ib]),('train/tune metadata mismatch',split,k)
        for k in fa:assert np.array_equal(fa[k][ia],fb[k][ib]),('train/tune features mismatch',split,k)
    ta=ma['split']==2;tb=mb['split']==2
    assert not set(ma['episode'][ta])&set(mb['episode'][tb]),'Heldout episodes overlap'
    m={k:np.concatenate([ma[k],mb[k][tb]]) for k in ['labels','episode','frame','episode_length','split']}
    f={k:np.concatenate([fa[k],fb[k][tb]]).astype(np.float64) for k in fa}
    del fa,fb
    names=ma['target_names'].tolist();train=np.flatnonzero(m['split']==0);tune=np.flatnonzero(m['split']==1);test=np.flatnonzero(m['split']==2)
    groups=[set(m['episode'][m['split']==s].tolist()) for s in range(3)]
    assert [len(g) for g in groups]==[10,3,40]
    assert all(not groups[i]&groups[j] for i in range(3) for j in range(i))
    assert len({(int(ep),int(fr)) for ep,fr in zip(m['episode'],m['frame'])})==len(m['split'])
    assert len(names)==22
    target=m['labels'].astype(np.float64); std=target[train].std(0);assert np.all(std>1e-8)
    split_columns=[names.index(n) for n in ['tool_block/cartesian_position[0]','tool_block/color[1]','floor_color[0]']]
    thresholds=np.median(target[train][:,split_columns],axis=0)
    bits=(target[:,split_columns]>thresholds).astype(np.int8)
    cells=bits@np.array([1,2,4],dtype=np.int8);parity=bits.sum(1)%2
    grid=f['grid'].reshape(-1,1280,4);haar=grid@H.T
    np.testing.assert_allclose(haar[:,:,0],2*f['mean'],rtol=1e-5,atol=2e-5)
    f['C']=haar[:,:,0];f['D']=haar[:,:,1:].reshape(-1,3840)
    def distribution(ix):
        z=target[ix][:,split_columns]
        return {'n':len(ix),'cell_counts':np.bincount(cells[ix],minlength=8).tolist(),
                'column_order':[names[j] for j in split_columns],
                'mean':z.mean(0).tolist() if len(z) else None,
                'std':z.std(0).tolist() if len(z) else None,
                'quantile_levels':[0.,.25,.5,.75,1.],
                'quantiles':np.quantile(z,[0.,.25,.5,.75,1.],axis=0).tolist() if len(z) else None}
    subsets={};selection={}
    for q in [0,1]:
        comp=train[parity[train]==q];rng=np.random.default_rng(SEED+q);parts=[];counts={}
        for ep in sorted(groups[0]):
            available=train[m['episode'][train]==ep];count=int(np.sum(m['episode'][comp]==ep))
            parts.append(rng.choice(available,size=count,replace=False));counts[str(ep)]=count
        iid=np.sort(np.concatenate(parts));tt=tune[parity[tune]==q]
        if len(tt)==0 or len(comp)<=128:raise ValueError(f'Insufficient selected train/tune samples parity={q}: train={len(comp)}, tune={len(tt)}')
        subsets[q]={'comp':comp,'iid':iid,'tune':tt}
        selection[str(q)]={'counts_per_train_episode':counts,'comp_train_indices':comp.tolist(),'iid_train_indices':iid.tolist(),
                            'tune_indices':tt.tolist(),'comp_cell_counts':np.bincount(cells[comp],minlength=8).tolist(),
                            'iid_cell_counts':np.bincount(cells[iid],minlength=8).tolist(),'tune_cell_counts':np.bincount(cells[tt],minlength=8).tolist()}
    for q in [0,1]:
        selection[str(q)]['distribution_audit']={
            'comp_train':distribution(subsets[q]['comp']),
            'iid_train':distribution(subsets[q]['iid']),
            'shared_tune':distribution(subsets[q]['tune']),
            'shared_test_all':distribution(test),
            'shared_test_ID':distribution(test[parity[test]==q]),
            'shared_test_OOD':distribution(test[parity[test]!=q])}
    config={'scope':'New exploratory analysis of two reused 20-sequence holdouts, not fresh confirmation or physical intervention',
            'split_columns':[names[i] for i in split_columns],'thresholds_all_original_train640':thresholds.tolist(),
            'cell_rule':'bits=(value>median), cell=xbit+2*greenbit+4*floorbit; parity=sum(bits)%2',
            'comparisons':'comp restricted to each parity; IID sampled per train episode with same count; both use identical parity-restricted tune',
            'normalization':'original full train640 target std shared by every condition',
            'models':['mean128','grid128','C128','D128','C64D64'],'rank_rule':'fixed128; fail if selected train sample count or numerical PCA rank insufficient; no adaptive rank',
            'pca':'only selected training subset; n-1 whitening; no tune/test in fitting',
            'alphas':ALPHAS,'selection':'supervised per-target tune alpha; train-only PCA; no test selection',
            'families':FAMILIES,'seed':SEED,'bootstrap':{'draws':10000,'confidence':.95,'unit':'paired sequence, equal weight; two parities reported separately using the same40 sequences, never80 independent sequences','scope':'exploratory conditional on fitted models, no multiplicity-confirmation claim'},
            'contrasts':'comp minus same-sized IID on identical OOD samples; ID counterpart; DID=(compOOD-iidOOD)-(compID-iidID), using sequences with both domains',
            'missing_domain':'sequence means computed only when at least one frame in domain; missing recorded, no zero imputation; DID on paired domain intersection',
            'selection_indices':selection,'inputs':{str(path/k):sha(path/k) for path in [a.input_a,a.input_b] for k in ['features.npz','sample_manifest.npz']},
            'marginal_matching':'IID matches sample counts within each training episode, not exact continuous label marginals; distribution audits retain means/std/quantiles',
            'source_sha256':sha(Path(__file__)),'helpers_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['complex_haar_evaluate.py','complex_spatial_evaluate.py']},
            'device':str(device),'torch':torch.__version__}
    write(a.output/'config.json',config)
    seq=np.unique(m['episode'][test]);tep=m['episode'][test]
    arrays={'targets':target,'target_names':names,'episode':m['episode'],'frame':m['frame'],'split':m['split'],'cells':cells,'bits':bits,'parity':parity,
            'thresholds':thresholds,'train_std':std,'original_train_indices':train,'original_tune_indices':tune,'test_indices':test,'test_source_cache':np.repeat([0,1],1280)}
    for q,sub in subsets.items():
        for k,v in sub.items():arrays[f'parity{q}_{k}_indices']=v
    np.savez_compressed(a.output/'audit_arrays.npz',**arrays)
    rows=[];predictions={};errors={};fits=[]
    for q in [0,1]:
        tt=subsets[q]['tune'];domain=[parity[test]==q,parity[test]!=q]
        counts=np.asarray([[int(np.sum((tep==ep)&mask)) for ep in seq] for mask in domain])
        for mode in ['comp','iid']:
            tr=subsets[q][mode];maps={};P={};label=f'p{q}_{mode}'
            for kind in ['mean','grid','C','D']:
                x=torch.as_tensor(f[kind][tr],device=device,dtype=torch.float64);mean=x.mean(0)
                _,s,v=torch.linalg.svd(x-mean,full_matrices=False)
                if len(s)<128 or float(s[127])<=float(s[0])*1e-8:
                    write(a.output/'FAILED.json',{'reason':'selected training subset numerical rank below128','condition':label,'kind':kind,'samples':len(tr)})
                    raise ValueError(f'Fixed-rank PCA failure: {label} {kind}')
                basis=v[:128].T/(s[:128]/np.sqrt(len(tr)-1))
                P[kind]=[((torch.as_tensor(f[kind][ix],device=device,dtype=torch.float64)-mean)@basis).cpu().numpy() for ix in [tr,tt,test]]
                maps[kind+'_mean']=mean.cpu().numpy();maps[kind+'_basis128']=basis.cpu().numpy();maps[kind+'_spectrum']=s.cpu().numpy()
                fits.append({'condition':label,'kind':kind,'train_count':len(tr),'variance_fraction128':float(s[:128].square().sum()/s.square().sum())})
            np.savez_compressed(a.output/(label+'_maps.npz'),**maps)
            models={k+'128':P[k] for k in ['mean','grid','C','D']}
            models['C64D64']=[np.concatenate([c[:,:64],d[:,:64]],1) for c,d in zip(P['C'],P['D'])]
            Y=[target[ix] for ix in [tr,tt,test]]
            for model,X in models.items():
                key=label+'_'+model;pred,fit=fit_readout(X,Y,device)
                np.savez_compressed(a.output/(key+'_fit.npz'),**fit)
                predictions[key]=pred;err=((pred-Y[2])/std)**2
                per=np.full((2,len(seq),len(names)),np.nan)
                for d,mask in enumerate(domain):
                    for j,ep in enumerate(seq):
                        selected=(tep==ep)&mask
                        if selected.any():per[d,j]=err[selected].mean(0)
                errors[key]=per
                family_scores={family:{dom:float(np.nanmean(per[d][:, [names.index(n) for n in targets]].mean(1))) for d,dom in enumerate(['ID','OOD'])} for family,targets in FAMILIES.items()}
                rows.append({'key':key,'parity':q,'mode':mode,'model':model,'train_count':len(tr),'tune_count':len(tt),
                             'alpha':fit['ratio_alpha_candidates'][fit['selected_candidate'],1].tolist(),'family_nmse':family_scores,
                             'domain_order':['ID','OOD'],'domain_sample_counts_per_sequence':counts.tolist(),
                             'nmse_per_target_by_domain':[[float(v) if np.isfinite(v) else None for v in np.nanmean(per[d],axis=0)] for d in range(2)]})
                print('completed',key,flush=True)
    contrasts=[]
    for q in [0,1]:
        for model in ['mean128','grid128','C128','D128','C64D64']:
            comp=errors[f'p{q}_comp_{model}'];iid=errors[f'p{q}_iid_{model}']
            for family,targets in FAMILIES.items():
                cols=[names.index(n) for n in targets];ce=comp[:,:,cols].mean(2);ie=iid[:,:,cols].mean(2)
                diff=ce-ie
                contrasts.append({'parity':q,'model':model,'family':family,
                                  'comp_minus_iid_ID':interval(diff[0],seq,SEED+q),
                                  'comp_minus_iid_OOD':interval(diff[1],seq,SEED+q),
                                  'difference_in_differences':interval(diff[1]-diff[0],seq,SEED+q),
                                  'comp_OOD_minus_ID':interval(ce[1]-ce[0],seq,SEED+q),
                                  'iid_OOD_minus_ID':interval(ie[1]-ie[0],seq,SEED+q)})
    np.savez_compressed(a.output/'predictions.npz',**predictions)
    np.savez_compressed(a.output/'per_sequence_nmse.npz',**errors,sequence_ids=seq,domain_names=['ID','OOD'])
    write(a.output/'results.json',{'scope':config['scope'],'targets':names,'sequence_ids':seq.tolist(),'rows':rows,'pca_fits':fits,'paired_contrasts':contrasts,
          'limitations':['Observational label-cell partition is not an intervention','IID may observe both parities; contrasts measure coverage effects conditional on this construction',
                         'Only one matched IID draw per parity; bootstrap does not capture training-set or tune uncertainty',
                         'Reused holdout and multiple exploratory comparisons; no fresh confirmation or unique factors claim',
                         'PCA fitted separately in comp/IID, so coverage changes representation fitting as well as readout fitting']})
    write(a.output/'COMPLETE.json',{'complete':True,'conditions':len(rows),'seconds':time.time()-started,'source_sha256':sha(Path(__file__))})


if __name__=='__main__':main()

