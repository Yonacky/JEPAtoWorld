"""G7f supervised ANOVA main-effect blocks plus shared interaction residual."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,time
import numpy as np
import torch
from fit_target_supervised_reference_20261012 import construct,sha,dump
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'runs/intervention_20261012';O=ROOT/'runs/family_anova_blocks_20261013'


def components(h):
    # h is [scene, factor0, factor1, factor2, latent]; each main has zero grid mean.
    grand=h.mean((1,2,3),keepdims=True)
    mains=[h.mean(tuple(j for j in (1,2,3) if j!=k),keepdims=True)-grand for k in (1,2,3)]
    mains=[np.broadcast_to(m,h.shape).copy() for m in mains]
    interaction=h-grand-sum(mains)
    np.testing.assert_allclose(grand+sum(mains)+interaction,h,atol=1e-12)
    for k,m in enumerate(mains):
        np.testing.assert_allclose(m.mean((1,2,3)),0,atol=1e-12)
        np.testing.assert_allclose((m*interaction).sum((1,2,3)),0,atol=1e-10)
    return mains+[interaction]


def projectors(q):return torch.stack([q[:,i*16:(i+1)*16]@q[:,i*16:(i+1)*16].T for i in range(4)])


def fit(C,q0):
    q=torch.tensor(q0,dtype=torch.float64,requires_grad=True);ct=torch.tensor(C);hist=[]
    def loss(q):return -torch.einsum('kij,kji->',projectors(q),ct)/4
    for step in range(301):
        cost=loss(q);hist.append({'step':step,'negative_mean_retention':float(cost.detach())})
        if step==300:break
        if q.grad is not None:q.grad.zero_()
        cost.backward()
        with torch.no_grad():
            g=q.grad;v=q.T@g;g=g-q@((v+v.T)/2);gn=float((g*g).sum());lr=1.
            accepted=False
            for _ in range(30):
                u,_,vh=torch.linalg.svd(q-lr*g);nq=u@vh
                if float(loss(nq))<=float(cost)-1e-4*lr*gn+1e-14:accepted=True;break
                lr*=.5
            if accepted:q.copy_(nq)
            hist[-1].update(lr=lr,accepted=accepted)
        if step%50==0:print('fit',step,hist[-1],flush=True)
    return q.detach().numpy(),hist


def evaluate(x,z,U,recs,models,domain):
    result={};audit=[]
    cache={}
    for rec in recs:
        sc=rec['scene']
        if sc not in cache:
            X=x[rec['ids']];H=X@U;n=(X*X).sum(1)
            cache[sc]=(X,H,np.maximum(n[:,None]+n[None,:]-2*X@X.T,0),z['codes'][rec['ids']])
    for name,P in models.items():
        vals=[];predictions=[]
        for r in recs:
            X,H,dist,c=cache[r['scene']];k=r['factor'];a=r['a'];t=r['target'];d=r['d'];v=r['v'];change=v@P[k]
            err=1+((change*change).sum(1)-2*(change*r['t']).sum(1))/r['den']
            distances=dist[a]+2*np.einsum('ni,nji->nj',change,H[a,None]-H[None])+(change*change).sum(1)[:,None]
            nearest=distances.argmin(1);other=[j for j in range(3) if j!=k]
            keep=np.all(c[nearest][:,other]==c[a][:,other],axis=1);correct=c[nearest,k]==c[d,k]
            absolute=err*r['den']
            vals.append([err.mean(),absolute.mean(),correct.mean(),keep.mean(),(correct&keep).mean()])
            # Explicit full-space reconstruction for pre-fixed first7 of every scene/factor.
            full=X[a[:7]]+change[:7]@U.T
            direct=((full-X[t[:7]])**2).sum(1)/r['den'][:7]
            audit.append(float(abs(direct-err[:7]).max()))
            predictions.append(np.column_stack([np.full(len(a),r['scene']),np.full(len(a),k),a,d,t,nearest,err,absolute,correct,keep]))
        m=np.asarray(vals).reshape(-1,3,5)
        np.savez_compressed(O/f'{domain}_{name}.npz',scene_metrics=m,predictions=np.concatenate(predictions))
        result[name]={'factor_means':m.mean(0).tolist(),'macro':m.mean((0,1)).tolist()}
    return {'metric_order':['relative_error','absolute_squared_error','target_correct','nontarget_preserved','all_correct'],'models':result,'fullspace_audit_maxabs':max(audit)}


def main():
    O.mkdir(exist_ok=False);torch.set_num_threads(2);start=time.time()
    paths=[R/'pilot/dataset.npz',R/'features/features.npz',R/'step2b/fitted.npz']
    with np.load(paths[0]) as f:z={k:f[k] for k in ['scene','split','codes','pairs']}
    x=np.load(paths[1])['grid'].astype(float);b=np.load(paths[2]);U=b['U'];h=x@U
    cfg={'mechanism':'train-only per-scene full-factorial ANOVA main effects + combined order2/3 interaction; full orthogonal four blocks',
         'rank_each':16,'dimension':64,'steps':300,'starts':1,'objective':'mean of four normalized component retained energies; exhaustive orthogonal complement penalizes cross-block allocation implicitly',
         'initial':'joint0 factors plus orthonormal residual','supervision':'known train intervention grid identities; not unsupervised or DSReg theorem',
         'edit':'two permitted references000/222, cross-scene donor; only target main block replaced, recipient interaction retained; no test truth used in edit',
         'selection':'fixed final300; tune/dev report only; no hyperparameter search','split_scenes':[list(range(32)),list(range(32,40)),list(range(40,48))],
         'scope':'old IJEPA visual-factor exploration; LeJEPA available cache only physical aliases, unsuitable for independent-axis construction',
         'inputs':{str(p):sha(p) for p in paths},'script_sha256':sha(Path(__file__))}
    dump(O/'config.json',cfg)
    for sc in range(48):
        ids=np.flatnonzero(z['scene']==sc);assert len(ids)==27
        np.testing.assert_array_equal(z['codes'][ids]@np.array([9,3,1]),np.arange(27))
        assert np.all(z['split'][ids]==(0 if sc<32 else 1 if sc<40 else 2))
    comp=components(h[:32*27].reshape(32,3,3,3,64))
    Cs=[]
    for arr in comp:
        # Scene-equal normalized energies, rather than high-amplitude scenes dominating.
        cov=[]
        for a in arr.reshape(32,27,64):cov.append(a.T@a/np.sum(a*a))
        Cs.append(np.mean(cov,axis=0))
    Cs=np.asarray(Cs);initial=b['joint_0'];cols=np.concatenate([np.linalg.eigh(p)[1][:,-16:] for p in initial],axis=1)
    residual=np.eye(64)-sum(initial);q0=np.column_stack([cols,np.linalg.eigh(residual)[1][:,-16:]])
    np.testing.assert_allclose(q0.T@q0,np.eye(64),atol=1e-8)
    q,hist=fit(Cs,q0);P=projectors(torch.tensor(q)).numpy();np.testing.assert_allclose(q.T@q,np.eye(64),atol=1e-10)
    models={'joint0':initial,'dsreg':b['dsreg'],'anova':P[:3],'no_edit':np.zeros_like(initial)}
    np.savez_compressed(O/'fitted.npz',Q=q,U=U,component_covariances=Cs,**models,interaction_projector=P[3]);dump(O/'fit.json',hist)
    results={};decomposition={}
    for domain,scenes in [('train',list(range(32))),('tune',list(range(32,40))),('dev',list(range(40,48)))]:
        recs,_,_,pairs=construct(x,z,U,scenes);dump(O/f'{domain}_pairs.json',pairs)
        results[domain]=evaluate(x,z,U,recs,models,domain)
        co=components(np.stack([h[z['scene']==sc].reshape(3,3,3,64) for sc in scenes]))
        table=[]
        for c in co:
            vals=[]
            for xx in c.reshape(len(scenes),27,64):vals.append([float(np.sum((xx@pp)**2)/np.sum(xx*xx)) for pp in P])
            table.append(np.mean(vals,axis=0).tolist())
        decomposition[domain]=table
        print(domain,results[domain]['models']['anova']['macro'],flush=True)
    dump(O/'results.json',results);dump(O/'component_allocation.json',decomposition)
    saved=np.load(O/'fitted.npz');checks=[]
    for domain in results:
        for name in models:
            p=np.load(O/f'{domain}_{name}.npz')['predictions'];v=[]
            for sc in np.unique(p[:,0]):
                v.append([[p[(p[:,0]==sc)&(p[:,1]==k),j].mean() for j in [6,7,8,9]] for k in range(3)])
            np.testing.assert_allclose(np.mean(v,axis=0),np.array(results[domain]['models'][name]['factor_means'])[:,:4],atol=1e-12)
            checks.append(domain+'/'+name)
    val={'status':'PASS','orthogonal_error':float(abs(q.T@q-np.eye(64)).max()),'fullspace_maxabs':max(v['fullspace_audit_maxabs'] for v in results.values()),'saved_prediction_reductions':checks,'monotone_objective':bool(np.max(np.diff([v['negative_mean_retention'] for v in hist]))<1e-12),'seconds':time.time()-start}
    dump(O/'independent_validation.json',val)
    dump(O/'manifest.json',{str(p):sha(p) for p in list(O.iterdir())+[Path(__file__),Path(__file__).with_name('fit_target_supervised_reference_20261012.py')] if p.is_file() and p.name!='manifest.json'})
    print(val,flush=True)


if __name__=='__main__':main()
