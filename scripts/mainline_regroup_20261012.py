"""B-3: fixed label-free temporal regrouping; supervised group diagnostics only."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import argparse, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
from scipy import linalg as la
ALPHAS=np.array([1e-6,1e-4,1e-2,.1,1.,10.,100.])
LAGS=[1,2,4,8]
SPLITS=['train','tune','dev']

def write(p,obj): p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def r2(y,p): return 1-((y-p)**2).sum(0)/((y-y.mean(0))**2).sum(0)
def load(root,ds):
    cp=root/'runs/e04_inputs/cache'/f'{ds}_ijepa_r0.npz'
    z=np.load(cp); X=[z[s].astype(float) for s in SPLITS]; S=[z['source_'+s] for s in SPLITS]
    ids,names=([4,5],['paddle_left_y','paddle_right_y']) if ds=='pong' else ([4,6,7,8],['cyl_0_active','cyl_2_active','cyl_3_active','cyl_4_active'])
    sp=root/f'experiments/e03_fp32_20261011/{ds}_factor_schema.json'
    schema=json.loads(sp.read_text()); idx={g['name']:g['slice'][0] for g in schema if g['type']=='continuous'}
    assert [idx[n] for n in names]==ids
    for s in S: assert np.all(np.diff(s)==1)
    assert all(S[i+1][0]-S[i][-1]>max(LAGS) for i in range(2))
    raw=[z['target_'+s][:,ids].astype(float) for s in SPLITS]
    m,sd=raw[0].mean(0),raw[0].std(0); H=la.helmert(len(ids),full=False).T
    zy=[(y-m)/sd for y in raw]; Y=[np.column_stack([y,y.mean(1),y@H]) for y in zy]
    targets=names+['common_standardized_mean']+[f'helmert_difference_{j}' for j in range(len(ids)-1)]
    audit={'dataset':ds,'cache_sha256':sha(cp),'schema_sha256':sha(sp),'split_sizes':[len(s) for s in S],'source_ranges':[[int(s[0]),int(s[-1])] for s in S],'source_hashes':[hashlib.sha256(s.tobytes()).hexdigest() for s in S],'episode_boundaries':'Unknown; consecutive source index does not exclude resets','train_target_mean':m.tolist(),'train_target_std':sd.tolist(),'helmert':H.tolist()}
    return X,S,Y,targets,audit

def affinity(a,source,seed):
    # Resampling weights apply to original pairs, never joined bootstrap endpoints.
    n,d=a.shape; weights=np.ones(n)
    if seed:
        rng=np.random.default_rng(20261012+seed); nb=(n+63)//64
        chosen=rng.integers(nb,size=nb); weights=np.zeros(n)
        for b in chosen: weights[b*64:min((b+1)*64,n)]+=1
    mean=np.average(a,axis=0,weights=weights); sd=np.sqrt(np.average((a-mean)**2,axis=0,weights=weights)); z=(a-mean)/np.maximum(sd,1e-12)
    aff=np.zeros((d,d)); pair_info=[]
    for lag in LAGS:
        valid=(source[lag:]-source[:-lag]==lag)
        w=np.minimum(weights[:-lag],weights[lag:])*valid
        c=(z[:-lag]*w[:,None]).T@z[lag:]/w.sum()
        aff+=(c*c+(c*c).T)/(2*len(LAGS))
        pair_info.append({'lag':lag,'valid_source_pairs':int(valid.sum()),'positive_weight_pairs':int((w>0).sum()),'pair_weight_sum':float(w.sum())})
    np.fill_diagonal(aff,0)
    return aff,pair_info

def partition(aff,k):
    # Deterministic equal-size greedy graph partition. No labels enter.
    remaining=list(range(len(aff))); groups=[]
    while remaining:
        strength=aff[np.ix_(remaining,remaining)].sum(1)
        first=remaining[int(strength.argmax())]; remaining.remove(first); g=[first]
        while len(g)<k:
            score=aff[np.ix_(remaining,g)].mean(1)
            nxt=remaining[int(score.argmax())]; remaining.remove(nxt); g.append(nxt)
        groups.append(sorted(g))
    return np.array(groups,dtype=int)

def fit_predict(A,Y,coords):
    ps=[]; coefs=[]; alphas=[]; losses=[]; means=[]
    for j,c in enumerate(coords):
        xx=[a[:,c] for a in A]; mx=xx[0].mean(0); my=Y[0][:,j].mean(); xc=[x-mx for x in xx]
        G=xc[0].T@xc[0]/len(xc[0]); rhs=xc[0].T@(Y[0][:,j]-my)/len(xc[0])
        e,v=la.eigh(G); ws=np.array([v@((v.T@rhs)/(np.maximum(e,0)+al)) for al in ALPHAS])
        ls=((xc[1]@ws.T+my-Y[1][:,j,None])**2).mean(0); b=int(ls.argmin()); w=ws[b]
        ps.append(xc[2]@w+my);coefs.append(w);alphas.append(ALPHAS[b]);losses.append(ls[b]);means.append(np.r_[mx,my])
    return np.array(ps).T,np.array(coefs),np.array(alphas),np.array(losses),np.array(means)

def ci(y,p,ref):
    rng=np.random.default_rng(614); vals=[]; n=len(y); nb=(n+63)//64
    for _ in range(400):
        ix=np.concatenate([np.arange(b*64,min((b+1)*64,n)) for b in rng.integers(nb,size=nb)])
        vals.append(r2(y[ix],p[ix])-r2(y[ix],ref[ix]))
    return np.percentile(vals,[2.5,97.5],axis=0).T.tolist()

def run(root,O):
    O.mkdir(parents=True,exist_ok=True)
    config={'scope':'Reused r0 exploratory B-3; not fresh confirmation','device':'local CPU only','python':platform.python_version(),'numpy':np.__version__,'code_sha256':sha(Path(__file__)),'lags':LAGS,'widths':[4,8],'candidates':{'4':32,'8':16},'alphas':ALPHAS.tolist(),'methods':['original','random_seed0','random_seed1','random_seed2','temporal_seed0','temporal_seed1','temporal_seed2'],'transforms':['white','sfa','full_block4'],'fit':'Symmetric mean squared lag correlation at lags 1,2,4,8; deterministic balanced greedy coordinate partition. Seed0 full train; seeds1/2 original-pair weighted 64-frame train block bootstrap.','selection':'Only train labels select one group per target by standardized covariance energy; tune chooses ridge; dev only scores. Unsupervised candidate construction, supervised semantic group selection.','budget':'Each method partitions all 128 coordinates once into 128/k groups of k; identical candidate count within width. k8 original is fixed adjacent block pairing, not best among all 496 block pairs.','bootstrap':'400 paired 64-frame dev-block replicates; conditional uncertainty on reused dev','prohibition':'No dev-based configuration or seed selection; report all.'}
    write(O/'config.json',config)
    rows=[]; selection=[]; audits=[]; stability=[]; comparisons=[]
    for ds in ['pong','pinball']:
        X,S,Y,targets,audit=load(root,ds)
        for transform in config['transforms']:
            tp=root/'runs/e04_inputs/transforms'/f'{ds}_ijepa_r0_{transform}.npz'; q=np.load(tp)['past_basis'].astype(float)
            assert q.shape==(128,128); err=float(np.max(np.abs(q.T@q-np.eye(128)))); assert err<1e-6
            A=[x@q for x in X]; yc=(Y[0]-Y[0].mean(0))/Y[0].std(0); ax=A[0]-A[0].mean(0)
            score=(ax.T@yc/len(ax))**2/np.maximum(np.mean(ax*ax,axis=0)[:,None],1e-12)
            aa=[]; pair=[]
            for seed in range(3):
                af,pi=affinity(A[0],S[0],seed); aa.append(af);pair.append(pi)
            np.savez_compressed(O/f'{ds}_{transform}_affinities.npz',affinity=np.array(aa))
            audits.append(dict(audit,transform=transform,transform_sha256=sha(tp),orthogonality_max_error=err,pair_audits=pair))
            for k in [4,8]:
                partitions={'original':np.arange(128).reshape(-1,k)}
                for seed in range(3): partitions[f'random_seed{seed}']=np.random.default_rng(470+seed).permutation(128).reshape(-1,k)
                for seed in range(3): partitions[f'temporal_seed{seed}']=partition(aa[seed],k)
                preds={}
                for method,groups in partitions.items():
                    assert groups.shape==(128//k,k) and np.array_equal(np.sort(groups.ravel()),np.arange(128))
                    gs=score[groups].sum(1); chosen=gs.argmax(0); coords=groups[chosen]
                    p,w,al,loss,means=fit_predict(A,Y,coords); rr=r2(Y[2],p);preds[method]=p
                    key=f'{ds}_{transform}_k{k}_{method}'
                    np.savez_compressed(O/f'{key}.npz',prediction=p,truth=Y[2],source_dev=S[2],groups=groups,selected_group=chosen,coordinates=coords,alpha=al,tune_mse=loss,coefficients=w,means=means,target_names=np.array(targets))
                    for j,t in enumerate(targets):
                        row={'dataset':ds,'transform':transform,'width':k,'method':method,'target':t,'dev_r2':float(rr[j]),'alpha':float(al[j]),'candidate_count':len(groups),'group':int(chosen[j]),'coordinates':coords[j].tolist(),'original_block_count':int(len(np.unique(coords[j]//4)))}
                        rows.append(row);selection.append(dict(row,train_group_scores=gs[:,j].tolist(),tune_mse=float(loss[j])))
                for method in partitions:
                    if method=='original':continue
                    comparisons.append({'dataset':ds,'transform':transform,'width':k,'method':method,'reference':'original','targets':targets,'delta_r2':(r2(Y[2],preds[method])-r2(Y[2],preds['original'])).tolist(),'paired_block_ci95':ci(Y[2],preds[method],preds['original'])})
                g0=partitions['temporal_seed0']; b0=np.empty(128,dtype=int)
                for i,g in enumerate(g0):b0[g]=i
                pair0=b0[:,None]==b0[None,:]; upper=np.triu(np.ones((128,128),dtype=bool),1)
                for seed in [1,2]:
                    g1=partitions[f'temporal_seed{seed}']; b1=np.empty(128,dtype=int)
                    for i,g in enumerate(g1):b1[g]=i
                    pair1=b1[:,None]==b1[None,:]
                    stability.append({'dataset':ds,'transform':transform,'width':k,'seed':seed,'within_group_pair_retention':float((pair0&pair1&upper).sum()/(pair0&upper).sum()),'chance_retention':(k-1)/127,'exact_group_fraction':len(set(map(tuple,map(sorted,g0)))&set(map(tuple,map(sorted,g1))))/len(g0)})
            print(ds,transform,'complete',flush=True)
        write(O/'results.json',rows);write(O/'selection.json',selection);write(O/'audits.json',audits);write(O/'stability.json',stability);write(O/'comparisons.json',comparisons)
    write(O/'COMPUTATION_COMPLETE.json',{'rows':len(rows),'prediction_files':84,'status':'computed; independent verification pending'})

def verify(root,O):
    rows=json.loads((O/'results.json').read_text(encoding='utf-8')); maximum=0.;metricerr=0.; count=0; group_checks=0
    for ds in ['pong','pinball']:
        X,S,Y,targets,_=load(root,ds)
        for transform in ['white','sfa','full_block4']:
            q=np.load(root/'runs/e04_inputs/transforms'/f'{ds}_ijepa_r0_{transform}.npz')['past_basis'].astype(float); A=[x@q for x in X]
            for k in [4,8]:
                for method in ['original']+[f'{m}_seed{s}' for m in ['random','temporal'] for s in range(3)]:
                    z=np.load(O/f'{ds}_{transform}_k{k}_{method}.npz');p=z['prediction'];groups=z['groups'];coords=z['coordinates']
                    assert np.array_equal(z['source_dev'],S[2]); assert np.array_equal(z['truth'],Y[2]);assert np.array_equal(np.sort(groups.ravel()),np.arange(128));assert len(groups)==128//k
                    # Independently rebuild train diagnostic score and confirm selected candidates.
                    ax=A[0]-np.mean(A[0],axis=0); yt=(Y[0]-Y[0].mean(0))/Y[0].std(0)
                    score=np.square(np.einsum('ni,nj->ij',ax,yt)/len(ax))/np.mean(ax**2,axis=0)[:,None]
                    assert np.array_equal(np.argmax(score[groups].sum(1),axis=0),z['selected_group']);group_checks+=1
                    for j,t in enumerate(targets):
                        c=coords[j];x=A[0][:,c];mx=x.mean(0);my=Y[0][:,j].mean();xc=x-mx;yy=Y[0][:,j]-my
                        # Independent direct normal-equation solve, rather than eigensolver.
                        ws=[np.linalg.solve(xc.T@xc+len(x)*a*np.eye(k),xc.T@yy) for a in ALPHAS]
                        losses=np.array([np.mean(((A[1][:,c]-mx)@w+my-Y[1][:,j])**2) for w in ws]);best=int(losses.argmin())
                        assert ALPHAS[best]==z['alpha'][j]
                        pp=(A[2][:,c]-mx)@ws[best]+my;maximum=max(maximum,float(np.max(np.abs(pp-p[:,j]))))
                        row=next(r for r in rows if r['dataset']==ds and r['transform']==transform and r['width']==k and r['method']==method and r['target']==t)
                        val=1-np.dot(Y[2][:,j]-pp,Y[2][:,j]-pp)/np.dot(Y[2][:,j]-Y[2][:,j].mean(),Y[2][:,j]-Y[2][:,j].mean());metricerr=max(metricerr,abs(val-row['dev_r2']));count+=1
    assert maximum<1e-9 and metricerr<1e-10
    audit={'status':'passed','prediction_target_arrays':count,'group_candidate_budget_checks':group_checks,'max_prediction_abs_error_direct_solve':maximum,'max_r2_abs_error':metricerr,'source_and_targets':'all exactly match original cache-derived dev','script_sha256':sha(Path(__file__))}
    write(O/'independent_validation.json',audit)
    files={str(p.relative_to(O)):sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['manifest.json','COMPLETE.json']}
    write(O/'manifest.json',files);total=sum(p.stat().st_size for p in O.iterdir() if p.is_file());assert total<100_000_000
    write(O/'COMPLETE.json',dict(audit,total_bytes=total,scope='Exploratory, no independent new data confirmation'))
    print(json.dumps(audit),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    (verify if args.verify else run)(args.root,args.output)
