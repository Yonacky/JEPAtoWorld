"""Exploratory A0/A1; fit sees X only, labels enter downstream probes only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse, hashlib, json, pathlib, platform, time
import numpy as np
import torch

ALPHAS=np.array([0,1e-7,1e-5,1e-3,.01,.1,1,10,100.])
def dump(p,o): p.write_text(json.dumps(o,indent=2),encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def shrink(z,t): return np.sign(z)*np.maximum(np.abs(z)-t,0)
def r2(y,p): return float(1-np.sum((y-p)**2)/max(np.sum((y-y.mean(0))**2),1e-15))
def metrics(y,p,kind):
    if kind=='categorical': return {'accuracy':float(np.mean(y.argmax(1)==p.argmax(1)))}
    out={'r2':r2(y,p),'rmse':float(np.sqrt(np.mean((y-p)**2)))}
    if kind=='angle':
        # Does not assume whether target is [sin,cos] or [cos,sin].
        cosine=np.sum(y*p,1)/np.maximum(np.linalg.norm(y,axis=1)*np.linalg.norm(p,axis=1),1e-12)
        out['angular_error_deg']=float(np.degrees(np.arccos(np.clip(cosine,-1,1))).mean())
        out['r2_definition']='joint encoded two-component R2, not scalar angular R2'
    return out

def probe(X,Y,k):
    ym=Y[0].mean(0);ys=Y[0].std(0);ys=np.maximum(ys,1e-8)
    yy=[(y-ym)/ys for y in Y]
    xm=X[0].mean(0);xs=np.maximum(X[0].std(0),1e-8)
    cross=((X[0]-xm)/xs).T@yy[0]/len(X[0])
    ids=np.argsort(np.sum(cross**2,1))[-k:]
    xx=[(x-xm)[:,ids] for x in X]
    gram=xx[0].T@xx[0]/len(xx[0]);rhs=xx[0].T@yy[0]/len(xx[0])
    best=None
    for a in ALPHAS:
        w=np.linalg.pinv(gram+a*np.eye(k))@rhs
        loss=float(np.mean((xx[1]@w-yy[1])**2))
        if best is None or loss<best[0]:best=(loss,a,xx[2]@w*ys+ym)
    return best[2],{'alpha':float(best[1]),'tune_mse':best[0],'coordinates':ids.tolist()}

def invariants(X,D):
    A=np.linalg.inv(D);z=[x@A for x in X]
    ym=np.sin(X[0][:,:1])+.2*X[0][:,1:2] # label-free nontrivial regression target
    xc=[x-X[0].mean(0) for x in X];zc=[v-z[0].mean(0) for v in z]
    g=xc[0].T@xc[0]/len(xc[0]);b=xc[0].T@ym/len(xc[0]);alpha=.1
    w=np.linalg.solve(g+alpha*np.eye(4),b)
    wz=np.linalg.solve(zc[0].T@zc[0]/len(zc[0])+alpha*A.T@A,zc[0].T@ym/len(zc[0]))
    err=float(np.max(np.abs(xc[2]@w-zc[2]@wz)))
    ols=float(np.max(np.abs(xc[2]@(np.linalg.pinv(xc[0])@ym)-zc[2]@(np.linalg.pinv(zc[0])@ym))))
    rec=float(np.max(np.abs(z[2]@D-X[2])))
    assert max(err,ols,rec)<1e-7,(err,ols,rec)
    return {'pullback_ridge_prediction_maxabs':err,'ols_prediction_maxabs':ols,'linear_reconstruction_maxabs':rec}

def fit(X,kind,seed,args,device):
    # X: samples x blocks x 4; labels are deliberately absent from signature.
    rng=np.random.default_rng(seed)
    init=np.stack([np.linalg.qr(rng.normal(size=(4,4)))[0] for _ in range(X.shape[1])])
    if kind=='random_nonorthogonal':
        right=np.stack([np.linalg.qr(rng.normal(size=(4,4)))[0] for _ in range(X.shape[1])]);raw=(init*np.array([.8,.95,1.05,1.2])[None,None,:])@right
        return raw/np.linalg.norm(raw,axis=2,keepdims=True),[]
    if kind in ['identity','random']:
        return np.tile(np.eye(4),(X.shape[1],1,1)) if kind=='identity' else init,[]
    t=torch.as_tensor(X,dtype=torch.float32,device=device)
    d=torch.nn.Parameter(torch.as_tensor(init,dtype=torch.float32,device=device))
    opt=torch.optim.Adam([d],lr=.015);history=[]
    for step in range(args.steps):
        ids=rng.choice(len(X),min(1024,len(X)),replace=False)
        x=t[ids];z=torch.einsum('nbd,bdk->nbk',x,torch.linalg.inv(d))
        mask=torch.zeros_like(z).scatter_(-1,torch.topk(z.abs(),args.active,dim=-1).indices,1.)
        c=z*mask # exactly matched active budget in training; piecewise differentiable
        rec=torch.einsum('nbd,bdk->nbk',c,d)
        mse=(rec-x).square().mean();l1=c.abs().mean();loss=mse+args.lam*l1
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():
            u,s,v=torch.linalg.svd(d,full_matrices=False)
            if kind=='orthogonal':d.copy_(u@v)
            else:
                # s in [.8,1.2], row normalization preserves cond <= (1.2/.8)^2.
                q=(u*s.clamp(.8,1.2).unsqueeze(-2))@v
                d.copy_(q/torch.linalg.vector_norm(q,dim=2,keepdim=True))
        if step%50==0 or step==args.steps-1:history.append({'step':step,'loss':float(loss),'mse':float(mse),'l1':float(l1)})
    return d.detach().cpu().numpy().astype(float),history

def synthetic(args,device,out):
    rows=[];params={}
    for seed in range(args.seeds):
        rng=np.random.default_rng(100+seed);s=rng.laplace(size=(6144,4));s/=s.std(0)
        q=np.linalg.qr(rng.normal(size=(4,4)))[0];ob=np.eye(4);ob[0,1]=.5;ob[2,3]=.4
        for case in ['orthogonal','oblique_correlated','nonlinear','missing']:
            target=s.copy()
            if case=='oblique_correlated':target[:,1]=.4*s[:,0]+s[:,1]
            x=target@q if case=='orthogonal' else target@ob@q
            if case=='nonlinear':x[:,0]+=.4*x[:,1]**2
            if case=='missing':x=s.copy();x[:,3]=rng.normal(size=len(x));x=x@q
            X=[x[:4096],x[4096:5120],x[5120:]];Y=[target[:4096],target[4096:5120],target[5120:]]
            mean=X[0].mean(0);scale=np.sqrt(np.mean((X[0]-mean)**2));X=[(v-mean)/scale for v in X]
            for kind in ['orthogonal','nonorthogonal']:
                D,h=fit(X[0][:,None,:],kind,seed,args,device);D=D[0];Z=[v@np.linalg.inv(D) for v in X]
                row={'case':case,'seed':seed,'method':kind,'condition':float(np.linalg.cond(D)),'history':h,**invariants(X,D),'factors':[]}
                for j in range(4):
                    yy=[y[:,j:j+1] for y in Y];p,meta=probe(Z,yy,1);pf,_=probe(Z,yy,4)
                    row['factors'].append({'factor':j,'top1_r2':r2(yy[2],p),'full_r2':r2(yy[2],pf),**meta})
                if case=='missing':
                    row['missing_factor_sanity_pass']=abs(row['factors'][3]['full_r2'])<.05
                    assert row['missing_factor_sanity_pass'],row
                if case in ['orthogonal','oblique_correlated']:
                    row['linear_information_sanity_pass']=all(v['full_r2']>.99 for v in row['factors'])
                    assert row['linear_information_sanity_pass'],row
                rows.append(row);params[f'{case}_{kind}_{seed}']=D
            print('synthetic',case,seed,flush=True)
    dump(out/'synthetic.json',{'source_sha256':sha(pathlib.Path(__file__)),'rows':rows,'interpretation':'Mechanism audit only; failures are retained. Sparse objective is not guaranteed to identify correlated or nonlinear factors. Missing factor is independent of every observation.'})
    np.savez_compressed(out/'synthetic_matrices.npz',**params)

def real(args,device,out):
    rows=[];checks=[];fits=[];arrays={};predictions={};manifest=[]
    for ds in ['pong','pinball']:
        cp=args.root/'runs/e04_inputs/cache'/f'{ds}_ijepa_r0.npz';bp=args.root/'runs/e04_inputs/transforms'/f'{ds}_ijepa_r0_full_block4.npz'
        sp=args.root/'experiments/e03_fp32_20261011'/f'{ds}_factor_schema.json'
        data=np.load(cp);basis=np.load(bp)['past_basis'].astype(float)
        X=[data[s].astype(float)@basis for s in ['train','tune','dev']]
        Y=[data['target_'+s].astype(float) for s in ['train','tune','dev']];groups=json.loads(sp.read_text())
        # Optional shared trim; default zero preserves existing split with 128-frame gaps.
        X=[v[args.gap:-args.gap] if args.gap else v for v in X];Y=[v[args.gap:-args.gap] if args.gap else v for v in Y]
        mean=X[0].mean(0);X=[v-mean for v in X]
        manifest.append({'dataset':ds,'files':{str(p.relative_to(args.root)):sha(p) for p in [cp,bp,sp]},'sizes':[len(v) for v in X],'gap':args.gap,'cache_keys':list(data.keys())})
        # Diagnostic label-selected parent is frozen across every transform.
        parents={}
        for g in groups:
            a,b=g['slice'];yy=Y[0][:,a:b];yy=(yy-yy.mean(0))/np.maximum(yy.std(0),1e-8)
            co=X[0].T@yy/len(yy);parents[g['name']]=int(np.argmax(np.sum(co**2,axis=1).reshape(-1,4).sum(1)))
        for seed in range(args.seeds):
            for kind in ['identity','random','random_nonorthogonal','orthogonal','nonorthogonal']:
                D,h=fit(X[0].reshape(len(X[0]),-1,4),kind,seed,args,device);key=f'{ds}_{kind}_{seed}'
                arrays[key]=D;arrays[ds+'_mean']=mean
                Z=[np.einsum('nbd,bdk->nbk',x.reshape(len(x),-1,4),np.linalg.inv(D)).reshape(len(x),-1) for x in X]
                cond=np.linalg.cond(D);assert np.max(cond)<2.251
                fits.append({'dataset':ds,'method':kind,'seed':seed,'history':h,'max_condition':float(cond.max()),'row_norm_error':float(np.max(abs(np.linalg.norm(D,axis=2)-1))),'dictionary_width_per_block':4,'stored_parameters_per_block':16,'effective_degrees_of_freedom':6 if kind=='orthogonal' else (12 if kind=='nonorthogonal' else 0)})
                for block in range(D.shape[0]):
                    xx=[x[:,block*4:block*4+4] for x in X];ck=invariants(xx,D[block]);checks.append({'key':key,'block':block,**ck})
                for variant in ['linear','shrink','top2']:
                    V=Z if variant=='linear' else [shrink(z,args.tau) for z in Z]
                    if variant=='top2':
                        V=[]
                        for z in Z:
                            zz=z.reshape(len(z),-1,4);ii=np.argsort(abs(zz),axis=2)[:,:,-args.active:];mask=np.zeros_like(zz);np.put_along_axis(mask,ii,1.,axis=2);V.append((zz*mask).reshape(len(z),-1))
                    rec=np.einsum('nbd,bdk->nbk',V[2].reshape(len(V[2]),-1,4),D).reshape(len(V[2]),-1)
                    fits.append({'key':key,'variant':variant,'dev_reconstruction_mse':float(np.mean((rec-X[2])**2)),'dev_active_per_block':float(np.mean(np.sum(abs(V[2].reshape(len(V[2]),-1,4))>1e-12,axis=2))),'linear_effective_rank_per_block':np.linalg.matrix_rank(D).tolist()})
                    for g in groups:
                        a,b=g['slice'];pa=parents[g['name']];xx=[v[:,pa*4:pa*4+4] for v in V];yy=[y[:,a:b] for y in Y]
                        for k in [1,2,4]:
                            p,m=probe(xx,yy,k);pk=f'{key}_{variant}_{g["name"]}_{k}';predictions[pk]=p
                            rows.append({'dataset':ds,'method':kind,'seed':seed,'variant':variant,'factor':g['name'],'type':g['type'],'parent':pa,'budget':k,**m,**metrics(yy[2],p,g['type'])})
                print(key,'condition',float(cond.max()),flush=True)
        predictions[ds+'_target_dev']=Y[2]
    dump(out/'results.json',{'source_sha256':sha(pathlib.Path(__file__)),'status':'exploratory_r0','rows':rows,'fits':fits,'checks':checks,'manifest':manifest,'limitations':['Existing r0 has been repeatedly explored; not confirmation.','Existing split gaps are retained; no new temporal pairs. Independence across trajectories is not established.','Nonorthogonal manifold has more degrees of freedom (12 vs 6); width and stored parameter shape are matched, intrinsic capacity is not.','Training top-k coding uses identical active budget; shrink is an extra diagnostic whose active counts need not match.','No local extension is authorized by this script.']})
    np.savez_compressed(out/'transforms.npz',**arrays);np.savez_compressed(out/'predictions.npz',**predictions)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True)
    ap.add_argument('--steps',type=int,default=400);ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--tau',type=float,default=.1);ap.add_argument('--lam',type=float,default=.03);ap.add_argument('--gap',type=int,default=0)
    ap.add_argument('--active',type=int,default=2);ap.add_argument('--stage',choices=['synthetic','real','all'],default='all');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu';start=time.time()
    dump(args.output/f'config_{args.stage}.json',{**{k:str(v) if isinstance(v,pathlib.Path) else v for k,v in vars(args).items()},'device':device,'torch':torch.__version__,'numpy':np.__version__,'python':platform.python_version(),'source_sha256':sha(pathlib.Path(__file__)),'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'objective':'mean top-k reconstruction error + lambda mean absolute top-k code; same k=2 budget, unit decoder atoms; no labels in fit'})
    if args.stage in ['synthetic','all']:synthetic(args,device,args.output)
    if args.stage in ['real','all']:real(args,device,args.output)
    dump(args.output/f'COMPLETE_{args.stage}.json',{'source_sha256':sha(pathlib.Path(__file__)),'seconds':time.time()-start,'status':'exploratory','stage':args.stage})
