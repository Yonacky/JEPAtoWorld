"""H-1: CPU-only old-r0 continuous-index history diagnostics, not confirmation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='2'
from pathlib import Path
import hashlib,json,platform,time
import numpy as np
from scipy import linalg
R=Path('F:/2610/CVPR27'); O=R/'runs/mainline_history_20261012'; O.mkdir(exist_ok=True)
A=np.array([1e-6,1e-4,.01,.1,1,10,100]); SPLITS=['train','tune','dev']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def r2(y,p): return 1-np.sum((y-p)**2,axis=0)/np.maximum(np.sum((y-y.mean(0))**2,axis=0),1e-14)
def boot(y,p,q,source):
    # Blocks never bridge a source discontinuity. Interval conditional on one old segment.
    cuts=np.r_[0,np.flatnonzero(np.diff(source)!=1)+1,len(source)]
    blocks=[np.arange(i,min(i+64,b)) for a,b in zip(cuts[:-1],cuts[1:]) for i in range(a,b,64)]
    rng=np.random.default_rng(20261012); vals=[]
    for _ in range(200):
        ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))]); vals.append(np.mean(r2(y[ix],p[ix])-r2(y[ix],q[ix])))
    return np.percentile(vals,[2.5,97.5]).tolist()
config={'scope':'Previously inspected r0, exploratory continuous-index windows only; NOT independent confirmation','device':'CPU','history':[1,2,4],'horizons':[1,2,4],'alphas':A.tolist(),'crop':'All variants use t>=3 and t+4 within same split, and every index t-3..t+4 consecutive','boundaries':'Episode/reset identities absent; no claim of true trajectory boundary audit','features':'Existing train-fitted whitened128; full history means all 128*h retained coordinates, not original1280','scaling':'Concatenation divided by sqrt(h), repeated-current same; train centered; no per-feature rescaling','PCA':'Train covariance only, top128, no labels or whitening; each method independently tune-selects ridge','selection':'Per dynamic component alpha; one common alpha per horizon minimizing tune mean squared error','bootstrap':'200 paired contiguous64-frame blocks, conditional on reused segment; not trajectories','code_sha256':sha(Path(__file__)),'python':platform.python_version(),'numpy':np.__version__}
write(O/'config.json',config)
rows=[]; audits=[]; comparisons=[]; validation=[]
for ds,ids,names in [('pong',[0,1],['ball_vel_dir_component0','ball_vel_dir_component1']),('pinball',[1,3],['ball_x_vel','ball_y_vel'])]:
    cp=R/f'runs/e04_inputs/cache/{ds}_ijepa_r0.npz'; sp=R/f'experiments/e03_fp32_20261011/{ds}_factor_schema.json'; schema=json.loads(sp.read_text()); z=np.load(cp)
    assert (ds!='pong' or schema[0]['name']=='ball_vel_dir')
    assert (ds!='pinball' or [g['name'] for g in schema if g['slice'][0] in ids]==names)
    X=[z[s] for s in SPLITS]; S=[z['source_'+s] for s in SPLITS]
    assert all(np.all(np.diff(s)>0) for s in S)
    assert all(not np.intersect1d(S[i],S[j]).size for i in range(3) for j in range(i))
    ix=[]
    for s in S:
        t=np.arange(3,len(s)-4); valid=np.all(np.stack([s[t+k]==s[t]+k for k in range(-3,5)]),axis=0); ix.append(t[valid])
    sources=[s[t] for s,t in zip(S,ix)]
    Y=[np.column_stack([z['target_'+s][t][:,ids]]+[x[t+h] for h in [1,2,4]]) for s,x,t in zip(SPLITS,X,ix)]
    groups=[slice(0,1),slice(1,2),slice(2,130),slice(130,258),slice(258,386)]
    np.savez_compressed(O/f'{ds}_indices.npz',**{s:sources[i] for i,s in enumerate(SPLITS)},**{'row_'+s:ix[i] for i,s in enumerate(SPLITS)})
    audits.append({'dataset':ds,'cache_sha256':sha(cp),'schema_sha256':sha(sp),'schema':schema,'source_hashes':{s:hashlib.sha256(v.tobytes()).hexdigest() for s,v in zip(SPLITS,S)},'window_counts':list(map(len,ix)),'source_ranges':[[int(s[0]),int(s[-1])] for s in sources],'split_gaps':[int(S[i+1][0]-S[i][-1]-1) for i in range(2)],'unknown_episode_boundaries':True,'target_names':names})
    preds={}
    for h,kind,pca in [(1,'history',False)]+[(h,k,p) for h in [2,4] for k in ['history','repeat'] for p in [False,True]]:
        name=f'{kind}{h}_'+('pca128' if pca else 'full')
        V=[np.concatenate([x[t-j if kind=='history' else t] for j in range(h)],axis=1)/np.sqrt(h) for x,t in zip(X,ix)]
        center=V[0].mean(0); V=[v-center for v in V]; basis=np.eye(128*h)
        if pca:
            ev,b=linalg.eigh(V[0].T@V[0]/len(V[0])); basis=b[:,-128:]; V=[v@basis for v in V]
        ym=Y[0].mean(0); G=V[0].T@V[0]/len(V[0]); C=V[0].T@(Y[0]-ym)/len(V[0]); ev,b=linalg.eigh(G); ev=np.maximum(ev,0); rhs=b.T@C
        weights=[b@(rhs/(ev[:,None]+a)) for a in A]
        losses=np.array([np.mean((V[1]@w+ym-Y[1])**2,axis=0) for w in weights]); chosen=np.empty(386,dtype=int)
        for sl in groups: chosen[sl]=np.argmin(losses[:,sl].mean(1))
        w=np.column_stack([weights[chosen[j]][:,j] for j in range(386)]); pred=V[2]@w+ym; preds[name]=pred
        # Independent direct solve of normal equations for each selected regularizer.
        maxerr=0.
        for ai in np.unique(chosen):
            cols=np.flatnonzero(chosen==ai); ww=linalg.solve(G+A[ai]*np.eye(len(G)),C[:,cols],assume_a='pos'); maxerr=max(maxerr,float(np.max(np.abs(V[2]@ww+ym[cols]-pred[:,cols]))))
        assert maxerr<1e-6,maxerr
        path=O/f'{ds}_{name}.npz'; np.savez_compressed(path,prediction=pred.astype('float32'),truth=Y[2].astype('float32'),alpha=A[chosen],tune_mse=losses,chosen=chosen,weight=w,feature_center=center,pca_basis=basis,target_center=ym,source_dev=sources[2])
        loaded=np.load(path); independent=1-np.square(loaded['prediction'].astype(float)-loaded['truth']).sum(0)/np.square(loaded['truth'].astype(float)-loaded['truth'].astype(float).mean(0)).sum(0)
        err=float(np.max(np.abs(independent-r2(Y[2],pred)))); assert err<1e-5
        validation.append({'dataset':ds,'method':name,'direct_solve_prediction_max_error':maxerr,'saved_float32_metric_max_error':err,'file_sha256':sha(path),'input_width':V[0].shape[1],'numerical_train_rank':int(np.sum(ev>max(ev.max()*1e-9,1e-12)))})
        for label,sl in [('dynamics',slice(0,2)),('future1',slice(2,130)),('future2',slice(130,258)),('future4',slice(258,386))]:
            rows.append({'dataset':ds,'method':name,'target':label,'dev_mean_r2':float(r2(Y[2][:,sl],pred[:,sl]).mean()),'alpha':A[chosen[sl]].tolist(),'input_width':V[0].shape[1]})
        print(ds,name,'dynamic R2',r2(Y[2][:,:2],pred[:,:2]),flush=True)
    for h in [2,4]:
        for suffix in ['full','pca128']:
            key=f'history{h}_{suffix}'; ref=f'repeat{h}_{suffix}'
            for label,sl in [('dynamics',slice(0,2)),('future1',slice(2,130)),('future2',slice(130,258)),('future4',slice(258,386))]:
                comparisons.append({'dataset':ds,'method':key,'reference':ref,'target':label,'delta_mean_r2':float(np.mean(r2(Y[2][:,sl],preds[key][:,sl])-r2(Y[2][:,sl],preds[ref][:,sl]))),'paired_block_ci95':boot(Y[2][:,sl],preds[key][:,sl],preds[ref][:,sl],sources[2])})
    for h in [2,4]:
        delta=float(np.max(np.abs(preds[f'repeat{h}_full']-preds['history1_full']))); assert delta<1e-6; validation.append({'dataset':ds,'repeat_full_vs_current_max_prediction_error':delta,'history':h})
write(O/'results.json',{'rows':rows,'comparisons':comparisons,'audit':audits})
write(O/'validation.json',{'status':'passed','checks':validation,'recompute':'Saved float32 predictions independently scored; eigen ridge independently recomputed via direct SPD solve for every chosen alpha','artifact_bytes':sum(p.stat().st_size for p in O.iterdir())})
lines=['H-1 历史与动力学最简实证；2026-10-12；CPU。','仅复用已反复查看的Pong/Pinball r0探索。episode/reset身份缺失，只完成连续索引和split审计，不能声称通过真实轨迹边界审计或独立确认。','1/2/4帧共同裁剪：train8185/tune1017/dev1017；窗口t-3..t+4逐索引连续且在split内。因素读出是带标签监督诊断；PCA仅训练特征拟合。正则仅tune选择，dev仅报告。','输入每帧128维是既有FP32编码器缓存的白化截断表示；full历史为256/512维，含容量增益；PCA128是同读出宽度比较，但历史PCA存储变换参数更多。重复帧也独立调参。','Pong目标是速度方向的两编码分量，不声称速度大小；Pinball目标为vx/vy。未来预测为直接预测h1/2/4表示，不是递归rollout或物理状态预测。']
for ds in ['pong','pinball']:
    lines.append('\n'+ds+' dev平均R²：方法 / 动力学 / h1表示 / h2表示 / h4表示')
    for name in dict.fromkeys(r['method'] for r in rows if r['dataset']==ds):
        rr=[r['dev_mean_r2'] for r in rows if r['dataset']==ds and r['method']==name]; lines.append(name+' / '+' / '.join(f'{v:.6f}' for v in rr))
    lines.append('历史减同宽复制当前帧：Δ平均R²及200次64帧配对块区间（旧片段条件区间，非跨轨迹确认）')
    for c in comparisons:
        if c['dataset']==ds: lines.append(f"{c['method']} {c['target']}: {c['delta_mean_r2']:.6f} [{c['paired_block_ci95'][0]:.6f}, {c['paired_block_ci95'][1]:.6f}]")
lines+=['\n核验：每种模型选中正则均用独立直接线性求解重算预测；落盘FP32预测独立重算R²；复制当前帧full与单帧预测数值等价。详见validation.json。','结论仅按上述同预算和全宽结果分别判断，不把PCA损失混作历史无信息；历史收益不等于稳定动力学子空间或物理因果解释。未做质量/摩擦/动作干预、跨轨迹复现、编码器比较，R2-5仍未完成。','证据目录：runs/mainline_history_20261012；配置、源码/缓存/schema/source哈希、索引、权重、PCA、预测、逐正则tune误差、结果及独立数值核验均保存。']
(R/'docs/mainline_history_report_20261012.txt').write_text('\n'.join(lines),encoding='utf-8')
print('DONE',sum(p.stat().st_size for p in O.iterdir()),flush=True)
