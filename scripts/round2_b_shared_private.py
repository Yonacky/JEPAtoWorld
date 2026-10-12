"""Exploratory route B; label-free transforms, supervised diagnostic readouts."""
import argparse, hashlib, json, os, platform
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
import scipy
from scipy import linalg as la
import torch

ALPHAS = [1e-6, 1e-4, 1e-2, .1, 1., 10., 100.]

def write(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def r2(y, p):
    den = ((y-y.mean(0))**2).sum(0)
    return np.where(den > 1e-12, 1-((y-p)**2).sum(0)/np.maximum(den,1e-12), np.nan)

def ridge(X, Y, device):
    # One independently tune-selected regularizer per diagnostic target.
    mx, my = X[0].mean(0), Y[0].mean(0)
    xx = [torch.as_tensor(x-mx, dtype=torch.float64, device=device) for x in X]
    yy = [torch.as_tensor(y-my, dtype=torch.float64, device=device) for y in Y]
    ev, v = torch.linalg.eigh(xx[0].T@xx[0]/len(xx[0]))
    rhs = v.T@(xx[0].T@yy[0]/len(xx[0]))
    preds, losses = [], []
    for a in ALPHAS:
        w = v@(rhs/(ev.clamp_min(0)[:,None]+a))
        losses.append(((xx[1]@w-yy[1])**2).mean(0))
        preds.append((xx[2]@w).cpu().numpy()+my)
    ls = torch.stack(losses).cpu().numpy()
    best = ls.argmin(0)
    pred = np.stack(preds)[best, :, np.arange(Y[0].shape[1])].T
    return pred, np.asarray(ALPHAS)[best], ls.min(0)

def blocks(n, size, rng):
    parts = [np.arange(i,min(i+size,n)) for i in range(0,n,size)]
    return np.concatenate([parts[i] for i in rng.integers(len(parts), size=len(parts))])

def paired_ci(y, p, ref, count, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(count):
        ix = blocks(len(y), 64, rng)
        vals.append(r2(y[ix],p[ix])-r2(y[ix],ref[ix]))
    return np.nanpercentile(vals,[2.5,97.5],axis=0).T.tolist()

def decomposition(x, source, fitidx):
    xm=x[fitidx].mean(0); xc=x-xm
    ev,v=la.eigh(xc[fitidx].T@xc[fitidx]/len(fitidx)); order=np.argsort(ev)[::-1];ev,v=ev[order],v[:,order]
    c=v[:,:1]; null=la.null_space(c.T)
    residual=xc@null
    # Only original neighboring rows whose source indices are consecutive;
    # bootstrap contributes weights, never concatenated false transitions.
    weights=np.bincount(fitidx,minlength=len(x))
    valid=(np.diff(source)==1)&(weights[:-1]>0)&(weights[1:]>0)
    d=np.diff(residual,axis=0)[valid]
    w=np.minimum(weights[:-1],weights[1:])[valid].astype(float)
    if len(d)<8: raise RuntimeError('Insufficient within-split adjacent train pairs')
    C=residual[fitidx].T@residual[fitidx]/len(fitidx)
    D=(d*w[:,None]).T@d/w.sum()
    ce,cv=la.eigh(C); keep=ce>max(ce.max()*1e-9,1e-12)
    whit=cv[:,keep]/np.sqrt(ce[keep])
    se,sv=la.eigh(whit.T@D@whit)
    slow=null@whit@sv
    # Euclidean orthonormalization makes concatenated rank and decoder budget explicit.
    slow,_=la.qr(slow,mode='economic')
    return xm,ev,v,c,slow, int(valid.sum())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='auto');ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--bootstrap',type=int,default=200)
    args=ap.parse_args();O=args.output;O.mkdir(parents=True,exist_ok=True)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(2)
    config={'scope':'Exploration on repeatedly reused r0; no independent confirmation', 'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'device':device,'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'python':platform.python_version(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'alphas':ALPHAS,'block_bootstrap_size':64,'bootstrap_unit':'contiguous time blocks, not independent trajectories','labels':'Only predefined targets and supervised diagnostic parent selection/readout; no transform labels','capacity':'PCA and hierarchical feature widths 2/4; rank1 common + rank(k-1) residual; both d*k stored loadings, same linear decoder budget; hierarchical temporal fitting uses extra train statistics','fit_seed0':'full train','fit_other_seeds':'contiguous-block train bootstrap'}
    write(O/'config.json',config)
    rows=[];audits=[];comparisons=[]
    for ds,ids,names in [('pong',[4,5],['paddle_left_y','paddle_right_y']),('pinball',[4,6,7,8],['cyl_0_active','cyl_2_active','cyl_3_active','cyl_4_active'])]:
        schema_path=args.root/f'experiments/e03_fp32_20261011/{ds}_factor_schema.json'
        schema=json.loads(schema_path.read_text())
        schema_indices={g['name']:g['slice'][0] for g in schema if g['type']=='continuous'}
        assert [schema_indices[n] for n in names]==ids, 'Schema does not match predeclared diagnostic group'
        I=args.root/'runs/e04_inputs'; cp=I/'cache'/f'{ds}_ijepa_r0.npz';z=np.load(cp)
        X=[z[s].astype(np.float64) for s in ['train','tune','dev']];S=[z['source_'+s] for s in ['train','tune','dev']]
        assert all(np.all(np.diff(s)>0) for s in S)
        assert not any(np.intersect1d(S[i],S[j]).size for i in range(3) for j in range(i))
        raw=[z['target_'+s][:,ids] for s in ['train','tune','dev']];ym,ys=raw[0].mean(0),raw[0].std(0);assert (ys>1e-8).all()
        H=la.helmert(len(ids),full=False).T
        Z=[(y-ym)/ys for y in raw];Y=[np.column_stack([y,y.mean(1),y@H]) for y in Z]
        yn=names+['common_standardized_mean']+[f'helmert_difference_{i}' for i in range(H.shape[1])]
        q=np.load(I/'transforms'/f'{ds}_ijepa_r0_full_block4.npz')['past_basis']
        cross=(X[0]-X[0].mean(0)).T@Z[0]/len(X[0]);score=np.array([np.sum((q[:,j:j+4].T@cross)**2) for j in range(0,128,4)])
        block=int(score.argmax());qb=q[:,4*block:4*block+4];qo=np.delete(q,np.s_[4*block:4*block+4],axis=1)
        ev0=z['eigenvalues'];pca=z['pca']; gram=pca.T@pca
        inverse_check=float(np.max(np.abs(gram*np.sqrt(ev0[:,None]*ev0[None,:])-np.eye(128))))
        if inverse_check>1e-5:raise RuntimeError(f'Unexpected whitening convention {inverse_check}')
        audit={'dataset':ds,'schema_sha256':hashlib.sha256(schema_path.read_bytes()).hexdigest(),'episode_boundaries':'Unknown; temporal adjacency is checked from source indices but resets cannot be excluded','cache_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'source_hashes':[hashlib.sha256(s.tobytes()).hexdigest() for s in S],'source_ranges':[[int(s[0]),int(s[-1])] for s in S],'split_sizes':[len(s) for s in S],'parent_block':block,'parent_selection':'supervised diagnostic, group training standardized cross covariance energy only','factors':names,'target_mean':ym.tolist(),'target_std':ys.tolist(),'target_train_correlation':np.corrcoef(Z[0].T).tolist(),'helmert':H.tolist(),'pca_inverse_metric_check':inverse_check,'input_cache_dtype':str(z['train'].dtype),'feature_origin':'FP32 encoder cache stored float64','white_cov_max_error':float(np.max(np.abs(np.cov(X[0],rowvar=False,bias=True)-np.eye(128))))}
        preds={}
        def evaluate(label, inputs, seed, norm, rank, kind='factor', targets=Y, target_names=yn, prediction_source=None):
            pred,aa,ll=ridge(inputs,targets,device); key=f'{ds}_{norm}_{label}_seed{seed}_{kind}'
            np.savez_compressed(O/(key+'.npz'),prediction=pred,truth=targets[2],alpha=aa,tune_mse=ll,source_dev=S[2] if prediction_source is None else prediction_source,target_source_dev=S[2] if prediction_source is None else prediction_source+1)
            rr=r2(targets[2],pred)
            for j,name in enumerate(target_names):rows.append({'dataset':ds,'normalization':norm,'method':label,'seed':seed,'rank':rank,'kind':kind,'target':name,'dev_r2':float(rr[j]),'alpha':float(aa[j])})
            if kind=='factor':preds[(norm,label,seed)]=pred
            return pred
        for name,inputs in [('full128',X),('parent4',[x@qb for x in X]),('outside124',[x@qo for x in X])]:
            evaluate(name,inputs,0,'white',inputs[0].shape[1])
            # Fixed-budget lightweight nonlinear diagnostic; centers/bandwidth use train features only.
            rng=np.random.default_rng(940)
            centers=inputs[0][rng.choice(len(inputs[0]),128,replace=False)]
            def distance(a,b):
                return np.maximum((a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T,0)
            dc=distance(centers,centers);bw=float(np.median(dc[np.triu_indices(128,1)]))
            rbf=[np.exp(-distance(a,centers)/max(bw,1e-12)) for a in inputs]
            evaluate(name+'_rbf128',rbf,0,'white',128)
            np.savez_compressed(O/f'{ds}_{name}_rbf128_map.npz',centers=centers,bandwidth=bw)
        for ref in ['parent4','outside124']:
            p=preds[('white','full128',0)];r=preds[('white',ref,0)]
            comparisons.append({'dataset':ds,'normalization':'white','seed':0,'method':'full128','reference':ref,'targets':yn,'delta_r2':(r2(Y[2],p)-r2(Y[2],r)).tolist(),'paired_block_bootstrap_ci95':paired_ci(Y[2],p,r,args.bootstrap,500)})
        stability={}
        for norm,XX in [('white',X),('unwhite128',[x*np.sqrt(ev0) for x in X])]:
            for seed in range(args.seeds):
                ix=np.arange(len(X[0])) if seed==0 else blocks(len(X[0]),64,np.random.default_rng(seed))
                xm,ev,v,c,slow,npairs=decomposition(XX[0],S[0],ix)
                maps={'common1':c}
                for k in [2,4]:
                    maps[f'flat_pca{k}']=v[:,:k];maps[f'hierarchy{k}']=np.column_stack([c,slow[:,:k-1]])
                    maps[f'residual{k-1}']=slow[:,:k-1]
                if seed==0:stability[norm]={name:D for name,D in maps.items()}
                meta={'eigenvalues':ev.tolist(),'top_eigengap_relative':float((ev[0]-ev[1])/max(ev[0],1e-12)),'train_pairs':npairs,'subspace_cos2_to_seed0':{name:(la.svdvals(la.orth(stability[norm][name]).T@la.orth(D))**2).tolist() for name,D in maps.items()}}
                # Sequential PCA on orthogonal complement must reproduce flat PCA subspace except degenerate spectra.
                N=la.null_space(c.T); rr=(XX[0]-xm)@N; _,rv=la.eigh(rr[ix].T@rr[ix]/len(ix)); seq=np.column_stack([c,N@rv[:,-3:]])
                meta['sequential_pca4_projector_error']=float(la.norm(seq@seq.T-v[:,:4]@v[:,:4].T))
                write(O/f'{ds}_{norm}_seed{seed}_decomposition.json',meta)
                for name,D in maps.items():
                    np.savez_compressed(O/f'{ds}_{norm}_{name}_seed{seed}_map.npz',mean=xm,loadings=D)
                    inp=[(x-xm)@D for x in XX];evaluate(name,inp,seed,norm,D.shape[1])
                    # All feature-future pairs independently formed inside each split.
                    valid=[np.flatnonzero(np.diff(s)==1) for s in S]
                    assert all(np.all(s[ii+1]==s[ii]+1) for s,ii in zip(S,valid))
                    futureX=[a[ii] for a,ii in zip(inp,valid)];futureY=[xx[ii+1] for xx,ii in zip(X,valid)]
                    evaluate(name,futureX,seed,norm,D.shape[1],kind='future_h1',targets=futureY,target_names=[f'white_feature_{j}' for j in range(128)],prediction_source=S[2][valid[2]])
                    recon=[a@D.T+xm for a in inp]
                    meta.setdefault('dev_reconstruction_relative_mse',{})[name]=float(np.sum((recon[2]-XX[2])**2)/np.sum((XX[2]-xm)**2))
                write(O/f'{ds}_{norm}_seed{seed}_decomposition.json',meta)
                for k in [2,4]:
                    for ref in [f'flat_pca{k}','common1']:
                        p=preds[(norm,f'hierarchy{k}',seed)];r=preds[(norm,ref,seed)]
                        comparisons.append({'dataset':ds,'normalization':norm,'seed':seed,'method':f'hierarchy{k}','reference':ref,'targets':yn,'delta_r2':(r2(Y[2],p)-r2(Y[2],r)).tolist(),'paired_block_bootstrap_ci95':paired_ci(Y[2],p,r,args.bootstrap,1000+seed)})
        audits.append(audit);write(O/'audits.json',audits);write(O/'results.json',rows);write(O/'paired_comparisons.json',comparisons)
        print(ds,'done',flush=True)
    report=['路线B探索审计与最简层级分解。r0被重复使用，不能作独立确认。','诊断公共量是训练标准化因素的平均，差异是预定Helmert零和对比；它们均未进入变换拟合。父块标签选择单列为监督诊断。','白化表示PCA的方差可能近乎完全退化；白化/去白化各自内部同容量比较。去白化仅恢复保留128维内的方差，不能检查丢弃1152维的信息。','公共分支rank1 PCA，残差分支SFA；维数2/4与平坦PCA一致，但时序方法使用额外训练统计，不是等计算量。','每个split独立构造h1相邻对，无跨split时间对。3次拟合为完整训练+2次64帧块重采样，配对64帧块bootstrap是条件于本探索区段的区间，不是跨轨迹确认。','标签读出alpha只在tune选；未按dev选方法或秩。结果不证明因果独立或共同物理原因。']
    for a in audits:report.append(f"{a['dataset']}: parent={a['parent_block']}, whitening_error={a['white_cov_max_error']:.3g}")
    (O/'report.txt').write_text('\n'.join(report),encoding='utf-8');write(O/'COMPLETE.json',{'status':'complete','rows':len(rows),'datasets':2,'seeds':args.seeds,'scope':'exploratory only'})

if __name__=='__main__':main()
