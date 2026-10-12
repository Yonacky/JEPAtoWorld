"""G4a fixed spatial compression, labels only diagnostic readouts."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy import linalg as la
import complex_route_comparison_20261012 as c
GROUPS={'object_position':[0,1,2],'object_green':[3],'floor':[7],'stage':[10]}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def load(root):
 I=root/'runs/spatial_holdout_20261011/features';z=np.load(I/'features.npz');m=dict(np.load(I/'sample_manifest.npz'));grid=z['grid'].astype(float);mean=z['mean'].astype(float);assert grid.shape==(2112,5120);assert np.max(np.abs(grid.reshape(-1,1280,4).mean(2)-mean))<2e-5;mask=[m['split']==i for i in range(3)];E=[m['episode'][s] for s in mask];assert [len(np.unique(e)) for e in E]==[10,3,20];assert all(not set(E[i])&set(E[j]) for i in range(3) for j in range(i));assert len(set(zip(m['episode'].tolist(),m['frame'].tolist())))==len(grid);wt=[c.ew(e) for e in E];Y=[m['labels'][s].astype(float) for s in mask];ym=wt[0]@Y[0];sd=np.sqrt(wt[0]@((Y[0]-ym)**2));return I,m,grid,mean,mask,E,wt,Y,ym,sd
def pca(x,w,k):
 mu=w@x;_,s,v=la.svd((x-mu)*np.sqrt(w[:,None]),full_matrices=False);assert s[k-1]>s[0]*1e-8;return v[:k].T,mu,float(np.sum(s[:k]**2)/np.sum(s**2))
def run(root,O):
 O.mkdir(parents=True,exist_ok=True);I,m,grid,mean,mask,E,wt,Y,ym,sd=load(root);train=grid[mask[0]];mu=wt[0]@train;W=wt[0];maps={};meta=[]
 P,_,v=pca(train,W,64);maps['global_grid64']=P;meta.append({'method':'global_grid64','local_retained_variances':[v]})
 parts=[];vars=[]
 for cell in range(4):
  P,_,v=pca(train[:,cell::4],W,16);B=np.zeros((5120,16));B[cell::4]=P;parts.append(B);vars.append(v)
 maps['cell16x4']=np.concatenate(parts,1);meta.append({'method':'cell16x4','local_retained_variances':vars})
 # Mean32 plus orthonormal Helmert spatial differences32; no labels and no per-channel reweighting.
 H=la.helmert(4,full=False).T;hi=np.einsum('ncd,dk->nck',train.reshape(-1,1280,4),H).reshape(len(train),-1);P,_,v=pca(hi,W,32);Bhi=np.einsum('dk,ckr->cdr',H,P.reshape(1280,3,32)).reshape(5120,32);Pm,_,vm=pca(mean[mask[0]],W,32);Bm=np.repeat((Pm/4)[:,None,:],4,axis=1).reshape(5120,32);maps['mean32_diff32']=np.concatenate([Bm,Bhi],1);meta.append({'method':'mean32_diff32','local_retained_variances':[vm,v]})
 P,_,v=pca(mean[mask[0]],W,64);maps['mean64']=np.repeat((P/4)[:,None,:],4,axis=1).reshape(5120,64);meta.append({'method':'mean64','local_retained_variances':[v]})
 cfg={'methods':list(maps),'rank':64,'definition':'Global grid PCA64; each cell PCA16 concatenated; mean32 plus orthonormal spatial Helmert-difference PCA32; meanPCA64. All map to64 then common train-output whitening.','groups':GROUPS,'readout':'full64 shared group alpha, train/tune/dev sequence equal; target train scale; no supervised feature selection','alphas':c.AL.tolist(),'scope':'Reused CausalWorld10/3/20 sequences, exploratory. Same5120 grid input for spatial routes; raw grid5120 vsmean1280 not capacity matched. Raw ridge not information bound.','input_hashes':{str(p):sha(p) for p in [I/'features.npz',I/'sample_manifest.npz',I/'features.json']},'code_sha256':sha(Path(__file__)),'dependency_sha256':sha(Path(c.__file__))};dump(O/'config.json',cfg);rows=[];reconstruction=[];params={'mean_grid':mu,'target_mean':ym,'target_sd':sd,'helmert':H};Ys=[(y-ym)/sd for y in Y];representations={}
 for name,B in maps.items():
  T=(train-mu)@B;cov=T.T@(T*W[:,None]);ev,V=la.eigh(cov);assert ev[0]>ev[-1]*1e-10;A=B@(V/np.sqrt(ev));Z=(grid-mu)@A;assert np.max(np.abs(Z[mask[0]].T@(Z[mask[0]]*W[:,None])-np.eye(64)))<1e-7;representations[name]=[Z[b] for b in mask];params[name]=A;params[name+'_prewhite']=B
  # Euclidean span projection is basis/whitening invariant; separate each cell and overall input energy.
  Q=la.orth(B);xc=grid-mu;rec=(xc@Q)@Q.T;err=(xc-rec).reshape(-1,1280,4);den=xc.reshape(-1,1280,4)
  for split in range(3):
   ee=np.sum(err[mask[split]]**2,axis=1);dd=np.sum(den[mask[split]]**2,axis=1);ratio=(wt[split]@ee)/(wt[split]@dd);reconstruction.append({'method':name,'split':split,'cell_normalized_reconstruction_error':ratio.tolist(),'overall_normalized_error':float(np.sum(wt[split]@ee)/np.sum(wt[split]@dd))})
  next(x for x in meta if x['method']==name)['output_cov_condition_before_whitening']=float(ev[-1]/ev[0]);next(x for x in meta if x['method']==name)['effective_rank']=64
 representations['raw_grid']=[grid[b] for b in mask];representations['raw_mean']=[mean[b] for b in mask];np.savez_compressed(O/'maps.npz',**params)
 for name,X in representations.items():
  dec=c.decomp(X,wt[0])
  for group,ids in GROUPS.items():
   p,beta,mx,my,alpha,loss=c.fit(X,[y[:,ids] for y in Ys],wt,dec);p=p*sd[ids]+ym[ids];truth=Y[2][:,ids];per=[]
   for e in np.unique(E[2]):
    b=E[2]==e;er=p[b]-truth[b];per.append({'episode':int(e),'normalized_mse':float(np.mean((er/sd[ids])**2)),'physical_mse':float(np.mean(er**2)),'vector_rmse':float(np.sqrt(np.mean(np.sum(er**2,1))))})
   namep=f'{name}_{group}.npz';np.savez_compressed(O/namep,prediction=p,truth=truth,episode=E[2],frame=m['frame'][mask[2]],coefficients=beta,input_mean=mx,output_mean=my,alpha=alpha,ids=ids);rows.append({'method':name,'group':group,'dimensions':X[0].shape[1],'normalized_mse':float(np.mean([x['normalized_mse'] for x in per])),'physical_mse':float(np.mean([x['physical_mse'] for x in per])),'per_episode':per,'alpha':alpha,'tune_mse':loss,'file':namep})
  print(name,'complete',flush=True)
 dump(O/'results.json',rows);dump(O/'compression_meta.json',meta);dump(O/'reconstruction.json',reconstruction);dump(O/'audit.json',{'episodes':[np.unique(e).tolist() for e in E],'rows':[int(b.sum()) for b in mask],'source_index_unique':True,'internal_reset_semantics':'Not verified; no dynamics claim','layout':'channel-major [N,1280,4], spatial cell order row-major','mean_grid_max_error':float(np.max(np.abs(grid.reshape(-1,1280,4).mean(2)-mean)))});dump(O/'COMPUTATION_COMPLETE.json',{'files':len(rows),'status':'computed'})
def verify(root,O):
 I,m,grid,mean,mask,E,wt,Y,ym,sd=load(root);maps=np.load(O/'maps.npz');rows=json.loads((O/'results.json').read_text());maximum=0.;metric=0.
 for row in rows:
  name=row['method'];X=[v[b] for b in mask for v in []] if False else None
  raw=grid if name=='raw_grid' else mean if name=='raw_mean' else (grid-maps['mean_grid'])@maps[name];X=[raw[b] for b in mask];p=np.load(O/row['file']);ids=p['ids'];mx=wt[0]@X[0];my=wt[0]@((Y[0][:,ids]-ym[ids])/sd[ids]);xx=(X[0]-mx)*np.sqrt(wt[0][:,None]);yy=((Y[0][:,ids]-ym[ids])/sd[ids]-my)*np.sqrt(wt[0][:,None]);al=float(p['alpha']);d=xx.shape[1]
  beta=xx.T@la.solve(xx@xx.T+al*np.eye(len(xx)),yy,assume_a='pos') if d>len(xx) else la.solve(xx.T@xx+al*np.eye(d),xx.T@yy,assume_a='pos');pred=((X[2]-mx)@beta+my)*sd[ids]+ym[ids];maximum=max(maximum,float(np.max(np.abs(pred-p['prediction']))));assert np.array_equal(p['truth'],Y[2][:,ids]) and np.array_equal(p['episode'],E[2]) and np.array_equal(p['frame'],m['frame'][mask[2]]);v=float(np.sum(wt[2]*np.mean(((pred-Y[2][:,ids])/sd[ids])**2,1)));metric=max(metric,abs(v-row['normalized_mse']))
 assert maximum<1e-7 and metric<1e-8
 out={'status':'passed','files':len(rows),'prediction_max_error':maximum,'metric_max_error':metric,'source_target_indices':'exact','method':'Independent direct primal/dual solve versus main thin SVD','code_sha256':sha(Path(__file__))};dump(O/'independent_validation.json',out);dump(O/'COMPLETE.json',out);print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--verify',action='store_true');a=p.parse_args();(verify if a.verify else run)(a.root,a.output)
