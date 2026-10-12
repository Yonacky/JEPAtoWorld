"""A-3: four-way exploratory sparse coordinates; CPU, existing r0 only."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, pathlib, time, itertools, types, platform
import numpy as np
import torch
from round2_a_global import fit, sha, dump, ALPHAS, r2, invariants
ROOT=pathlib.Path('F:/2610/CVPR27'); OUT=ROOT/'runs/mainline_fourway_20261012'

def kmeans(x,seed):
    rng=np.random.default_rng(seed); c=x[rng.choice(len(x),4,replace=False)].copy()
    for _ in range(60):
        ids=((x[:,None]-c)**2).sum(2).argmin(1)
        nc=np.array([x[ids==j].mean(0) if np.any(ids==j) else x[rng.integers(len(x))] for j in range(4)])
        if np.max(abs(c-nc))<1e-8:break
        c=nc
    return c

def gate(x,c):return ((x[:,None]-c)**2).sum(2).argmin(1)
def align(z,ref):
    # Only permutation/sign alignment: preserves orthogonality and sparsity budget.
    corr=z.T@ref / np.maximum(np.linalg.norm(z,axis=0)[:,None]*np.linalg.norm(ref,axis=0)[None,:],1e-12)
    p=max(itertools.permutations(range(4)),key=lambda p:sum(abs(corr[p[j],j]) for j in range(4)))
    P=np.zeros((4,4))
    for j in range(4):P[p[j],j]=1 if corr[p[j],j]>=0 else -1
    return P

def probe(x,y,k):
    xm=x[0].mean(0); xs=np.maximum(x[0].std(0),1e-8);ym=y[0].mean(0)
    cor=((x[0]-xm)/xs).T@(y[0]-ym)/len(y[0]);ids=np.argsort(np.sum(cor**2,axis=1))[-k:]
    xx=[(v-xm)[:,ids] for v in x]; yy=y[0]-ym
    gram=xx[0].T@xx[0]/len(xx[0]);rhs=xx[0].T@yy/len(xx[0]);best=None
    for a in ALPHAS:
        w=np.linalg.pinv(gram+a*np.eye(k))@rhs; loss=float(np.mean((xx[1]@w+ym-y[1])**2))
        if best is None or loss<best[0]:best=(loss,float(a),w)
    return xx[2]@best[2]+ym, {'ids':ids.tolist(),'xm':xm.tolist(),'ym':ym.tolist(),'w':best[2].tolist(),'alpha':best[1],'tune_mse':best[0]}

def main():
    start=time.time();OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    args=types.SimpleNamespace(steps=200,active=2,lam=.03)
    cfg={'device':'cpu','seeds':[0,1,2],'steps_per_dictionary':200,'centers':4,'input_dim':4,'training_active':2,'lambda':.03,'train_batch_max':1024,'dtype_fit':'float32','dtype_checks':'float64','selection':'No label transform/gate fit. Train-label complete-parent and coordinate diagnostic; tune ridge only. No method selection on dev.','alignment':'Train activation correlation signed permutation to learned global orthogonal, all 24 permutations; no labels.','python':platform.python_version(),'torch':torch.__version__,'source_sha256':sha(pathlib.Path(__file__)),'fit_source_sha256':sha(ROOT/'scripts/round2_a_global.py')}
    dump(OUT/'config.json',cfg)
    rows=[];fits=[];manifest=[];params={};preds={};recompute=[]
    for ds,names in [('pong',['ball_x','ball_y']),('pinball',['cyl_0_active','cyl_3_active'])]:
        cp=ROOT/f'runs/e04_inputs/cache/{ds}_ijepa_r0.npz';bp=ROOT/f'runs/e04_inputs/transforms/{ds}_ijepa_r0_full_block4.npz';sp=ROOT/f'experiments/e03_fp32_20261011/{ds}_factor_schema.json'
        data=np.load(cp);basis=np.load(bp)['past_basis'].astype(float);schema=json.loads(sp.read_text())
        raw=[data[s].astype(float)@basis for s in ['train','tune','dev']];targets=[data['target_'+s].astype(float) for s in ['train','tune','dev']]
        manifest.append({'dataset':ds,'files':{str(p.relative_to(ROOT)):sha(p) for p in [cp,bp,sp]},'sizes':[len(v) for v in raw],'split':'existing r0 contiguous train/tune/dev with previously audited 128 frame gaps; episode boundaries unavailable'})
        for name in names:
            g=next(g for g in schema if g['name']==name);a,b=g['slice'];y=[t[:,a:b] for t in targets];yc=(y[0]-y[0].mean(0))/np.maximum(y[0].std(0),1e-8)
            cov=raw[0].T@yc/len(yc);parent=int(np.sum(cov**2,axis=1).reshape(-1,4).sum(1).argmax());raw4=[v[:,4*parent:4*parent+4] for v in raw]
            mean=raw4[0].mean(0);scale=float(np.sqrt(np.mean((raw4[0]-mean)**2)));x=[(v-mean)/scale for v in raw4]
            base=f'{ds}_{name}';params[base+'_mean']=mean;params[base+'_scale']=np.array(scale);params[base+'_parent']=np.array(parent);preds[base+'_target']=y[2]
            for seed in range(3):
                centers=kmeans(x[0],seed+314);gates=[gate(v,centers) for v in x];ref,h=fit(x[0][:,None,:],'orthogonal',seed,args,'cpu');ref=ref[0]
                for method in ['identity','global_orthogonal','global_nonorthogonal','local_orthogonal','local_nonorthogonal','gate_only']:
                    key=f'{base}_s{seed}_{method}';local=method.startswith('local');gateonly=method=='gate_only';hist=[]
                    if gateonly:dd=np.empty((0,4,4));z=[np.eye(4)[gg] for gg in gates]
                    else:
                        if method=='identity':dd=np.eye(4)[None]
                        elif method=='global_orthogonal':dd=ref[None];hist=[h]
                        elif not local:
                            dd,hh=fit(x[0][:,None,:],'nonorthogonal',seed,args,'cpu');hist=[hh]
                        else:
                            dd=[]
                            for j in range(4):
                                subset=x[0][gates[0]==j];d,hh=fit(subset[:,None,:],method.split('_',1)[1],seed,args,'cpu');d=d[0]
                                P=align(subset@np.linalg.inv(d),subset@np.linalg.inv(ref));dd.append(P.T@d);hist.append(hh)
                            dd=np.array(dd)
                        z=[np.einsum('ni,nij->nj',v,np.linalg.inv(dd)[gg if local else np.zeros(len(v),int)]) for v,gg in zip(x,gates)]
                    params[key+'_D']=dd;params[key+'_centers']=centers
                    checks=[]
                    if not gateonly:
                        for j,d in enumerate(dd):
                            xx=[v[gg==j] if local else v for v,gg in zip(x,gates)]
                            checks.append(invariants(xx,d))
                        assert np.linalg.cond(dd).max()<2.251
                        if 'orthogonal' in method and 'nonorthogonal' not in method: assert np.max(abs(dd@dd.transpose(0,2,1)-np.eye(4)))<5e-6
                    dist=np.sort(((x[2][:,None]-centers)**2).sum(2),axis=1);margin=(dist[:,1]-dist[:,0])/np.maximum(dist[:,1],1e-12)
                    boundary=margin<.1
                    fr={'key':key,'parent':parent,'stored_matrix_parameters':int(dd.size),'stored_gate_parameters':16 if local or gateonly else 0,'train_gate_counts':np.bincount(gates[0],minlength=4).tolist(),'tune_gate_counts':np.bincount(gates[1],minlength=4).tolist(),'dev_gate_counts':np.bincount(gates[2],minlength=4).tolist(),'max_condition':float(np.linalg.cond(dd).max()) if len(dd) else None,'rank':np.linalg.matrix_rank(dd).tolist() if len(dd) else [],'checks':checks,'history':hist,'boundary_dev_count':int(boundary.sum())}
                    if local:
                        # Coordinate mismatch between nearest and second nearest experts at same observed point.
                        near=np.argsort(((x[2][:,None]-centers)**2).sum(2),axis=1)[:,:2]
                        za=[np.einsum('ni,nij->nj',x[2],np.linalg.inv(dd)[near[:,j]]) for j in range(2)]
                        fr['boundary_aligned_coordinate_mismatch_mse']=float(np.mean((za[0][boundary]-za[1][boundary])**2)) if boundary.any() else None
                    for variant in (['gate'] if gateonly else ['linear','top2']):
                        zz=z
                        if variant=='top2':
                            zz=[]
                            for v in z:
                                mask=np.zeros_like(v);np.put_along_axis(mask,np.argsort(abs(v),axis=1)[:,-2:],1,axis=1);zz.append(v*mask)
                        if not gateonly:
                            rec=np.einsum('ni,nij->nj',zz[2],dd[gates[2] if local else np.zeros(len(x[2]),int)])
                            fr[variant+'_dev_reconstruction_mse']=float(np.mean((rec-x[2])**2))
                            fr[variant+'_boundary_reconstruction_mse']=float(np.mean((rec[boundary]-x[2][boundary])**2)) if boundary.any() else None
                        for k in ([4] if gateonly else [1,2,4]):
                            pred,meta=probe(zz,y,k);pk=f'{key}_{variant}_k{k}';preds[pk]=pred
                            again=(zz[2]-np.array(meta['xm']))[:,meta['ids']]@np.array(meta['w'])+np.array(meta['ym']);err=float(np.max(abs(again-pred)));assert err<1e-10
                            row={'key':pk,'dataset':ds,'factor':name,'parent':parent,'seed':seed,'method':method,'variant':variant,'budget':k,'dev_r2':r2(y[2],pred),'dev_mse':float(np.mean((y[2]-pred)**2)),'boundary_dev_mse':float(np.mean((y[2][boundary]-pred[boundary])**2)) if boundary.any() else None,**meta}
                            rows.append(row);recompute.append(err)
                    fits.append(fr)
                print(base,seed,'done',flush=True)
    np.savez_compressed(OUT/'transforms.npz',**params);np.savez_compressed(OUT/'predictions.npz',**preds)
    dump(OUT/'results.json',{'rows':rows,'fits':fits,'manifest':manifest,'prediction_recompute_maxabs':max(recompute),'status':'exploratory reused r0; no independent confirmation','limitations':['Four local matrices have 64 matrix parameters vs global 16; local gate adds16; local compute 4x200 steps, no equal total capacity/compute claim.','Hard routing, with only signed permutation coordinate alignment. Continuity not guaranteed.','Gate-only has four input coordinates and effective rank3, only full budget4; diagnostic not equal k1/k2 budget.','Training label parent selection restricts interpretation to diagnostic chosen input, not unsupervised whole-representation discovery.','No temporal dynamics, trajectory bootstrap, independent test, or high rank information claim.']})
    # Reload independently: stored prediction metrics, finite arrays and hashes.
    saved=np.load(OUT/'predictions.npz');rr=json.loads((OUT/'results.json').read_text());metricerr=[]
    for r in rr['rows']:
        y=saved[f"{r['dataset']}_{r['factor']}_target"];p=saved[r['key']];metricerr.append(abs(r2(y,p)-r['dev_r2']))
    assert max(metricerr)<1e-12
    dump(OUT/'validation.json',{'prediction_metric_reload_maxabs':max(metricerr),'prediction_weight_recompute_maxabs':max(recompute),'all_prediction_arrays_finite':all(np.isfinite(saved[k]).all() for k in saved.files),'checks_count':sum(len(f['checks']) for f in fits),'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()},'seconds':time.time()-start})
    dump(OUT/'COMPLETE.json',{'status':'calculation and numerical validation complete; exploratory only','seconds':time.time()-start})
if __name__=='__main__':main()
