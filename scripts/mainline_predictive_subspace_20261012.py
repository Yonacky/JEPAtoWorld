"""B-4: train-only temporal low-rank subspace; reused-r0 mechanism diagnostic."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
from pathlib import Path
import argparse,json,hashlib,platform
import numpy as np
from scipy import linalg as la
from mainline_regroup_20261012 import load,write,sha,r2,ALPHAS,LAGS

def weights(n,seed):
 w=np.ones(n)
 if seed:
  w[:]=0;rng=np.random.default_rng(7020+seed)
  for b in rng.integers((n+63)//64,size=(n+63)//64):w[b*64:min((b+1)*64,n)]+=1
 return w

def subspaces(x,s,ev,seed):
 w=weights(len(x),seed);m=np.average(x,axis=0,weights=w);xc=x-m;Cs=[];pa=[]
 for lag in LAGS:
  valid=s[lag:]-s[:-lag]==lag;ww=np.minimum(w[:-lag],w[lag:])*valid
  Cs.append((xc[:-lag]*ww[:,None]).T@xc[lag:]/ww.sum())
  pa.append({'lag':lag,'valid_pairs':int(valid.sum()),'positive_weight_pairs':int((ww>0).sum())})
 M=np.concatenate(Cs,axis=1)/np.sqrt(len(LAGS));U,sv,_=la.svd(M,full_matrices=False)
 # First temporal component is only a dominant predictive direction, not a semantic common factor.
 N=la.null_space(U[:,:1].T);Mr=np.concatenate([N.T@C@N for C in Cs],axis=1)/np.sqrt(len(LAGS));ur,sr,_=la.svd(Mr,full_matrices=False);R=N@ur
 ux=xc*np.sqrt(ev);cov=(ux*w[:,None]).T@ux/w.sum();pe,pv=la.eigh(cov);pv=pv[:,::-1];P,_=la.qr(np.sqrt(ev)[:,None]*pv,mode='economic')
 rand,_=la.qr(np.random.default_rng(900+seed).normal(size=(128,128)))
 return {'predictive':U,'residual_predictive':R,'pca_unwhite':P,'random':rand},sv,sr,M,pa

def ridge(A,Y):
 mx=A[0].mean(0);my=Y[0].mean(0);xc=[a-mx for a in A];yc=Y[0]-my
 e,v=la.eigh(xc[0].T@xc[0]/len(xc[0]));rhs=v.T@(xc[0].T@yc/len(xc[0]));ws=[v@(rhs/(np.maximum(e,0)[:,None]+a)) for a in ALPHAS]
 ls=np.array([np.mean((xc[1]@w+my-Y[1])**2,axis=0) for w in ws]);b=ls.argmin(0);w=np.column_stack([ws[b[j]][:,j] for j in range(Y[0].shape[1])]);p=xc[2]@w+my
 return p,w,mx,my,ALPHAS[b],ls.min(0)

def future_fit(X,B,S):
 # Common t crop for all four future horizons, inside each original split.
 A=[x[:-8]@B for x in X];Y=[np.concatenate([x[h:len(x)-8+h] for h in LAGS],axis=1)/2 for x in X]
 for s in S:
  for h in LAGS:assert np.all(s[h:len(s)-8+h]-s[:-8]==h)
 mx=A[0].mean(0);my=Y[0].mean(0);xc=[a-mx for a in A];e,v=la.eigh(xc[0].T@xc[0]/len(xc[0]));rhs=v.T@(xc[0].T@(Y[0]-my)/len(xc[0]));ws=[v@(rhs/(np.maximum(e,0)[:,None]+a)) for a in ALPHAS]
 losses=[np.mean((xc[1]@w+my-Y[1])**2) for w in ws];best=int(np.argmin(losses));w=ws[best];p=xc[2]@w+my;se=np.mean((p-Y[2])**2,axis=1);be=np.mean((Y[2]-my)**2,axis=1)
 return {'future_coefficients':w,'future_input_mean':mx,'future_target_mean':my,'future_alpha':ALPHAS[best],'future_dev_mse_per_row':se,'future_baseline_mse_per_row':be,'future_source':S[2][:-8]},float(se.mean()/be.mean())

def run(root,O):
 O.mkdir(parents=True,exist_ok=True);config={'scope':'B-4 mechanism exploration on reused r0, no fresh confirmation','device':'CPU','lags':LAGS,'ranks':[4,8],'seeds':[0,1,2],'seed0':'full train','other_seeds':'64-frame train blocks, weighted original endpoint pairs, no concatenated transitions','predictive':'Left SVD of concatenated lag cross-covariances in frozen train-whitened cache; no factor labels; no rewhitening on bootstrap','residual':'Project both past and future onto orthogonal complement of top predictive input direction; recompute SVD; this is not identified semantic common component','pca':'Original variance PCA inside retained128; fit weighted covariance of unwhitened retained coordinates and map span back to white Euclidean orthonormal basis','diagnostics':'Factor labels only train group selection/readout; tune ridge; dev scores. global maps are single label-free shared subspaces. supervised_best_original and supervised_best_spectral use 32/16 candidates per target; not identical discovery/search budgets to a single map.','future':'Multi-lag full-whitened-feature readout with single tune MSE selected alpha; save factored predictor and per-row error rather than full512 output','alphas':ALPHAS.tolist(),'python':platform.python_version(),'numpy':np.__version__,'code_sha256':sha(Path(__file__)),'dependency_sha256':sha(Path(__file__).with_name('mainline_regroup_20261012.py'))}
 write(O/'config.json',config);rows=[];audits=[];stability=[];energy=[];frows=[];association=[]
 for ds in ['pong','pinball']:
  X,S,Y,targets,audit=load(root,ds);z=np.load(root/f'runs/e04_inputs/cache/{ds}_ijepa_r0.npz');ev=z['eigenvalues'];q=np.load(root/f'runs/e04_inputs/transforms/{ds}_ijepa_r0_full_block4.npz')['past_basis'];audit['transform_sha256']=sha(root/f'runs/e04_inputs/transforms/{ds}_ijepa_r0_full_block4.npz');audit['white_cov_error']=float(np.max(np.abs(np.cov(X[0],rowvar=False,bias=True)-np.eye(128))));assert audit['white_cov_error']<1e-5
  base={}
  def evaluate(name,seed,k,maps,selection=None,future=False):
   pp=[];WW=[];MX=[];MY=[];AL=[]
   for j in range(Y[0].shape[1]):
    B=maps[j];a=[x@B for x in X];p,w,mx,my,al,_=ridge(a,[y[:,j:j+1] for y in Y]);pp.append(p[:,0]);WW.append(w[:,0]);MX.append(mx);MY.append(my[0]);AL.append(al[0])
   p=np.array(pp).T;rr=r2(Y[2],p);data={'prediction':p,'truth':Y[2],'source_dev':S[2],'maps':np.array(maps),'coefficients':np.array(WW),'input_means':np.array(MX),'target_means':np.array(MY),'alpha':np.array(AL),'targets':np.array(targets)}
   if selection is not None:data['selected_group']=selection
   if future:
    ff,nmse=future_fit(X,maps[0],S);data.update(ff);frows.append({'dataset':ds,'method':name,'seed':seed,'rank':k,'future_nmse':nmse})
   np.savez_compressed(O/f'{ds}_{name}_s{seed}_k{k}.npz',**data)
   for j,t in enumerate(targets):rows.append({'dataset':ds,'method':name,'seed':seed,'rank':k,'target':t,'dev_r2':float(rr[j]),'alpha':float(AL[j]),'selected_group':None if selection is None else int(selection[j])})
  evaluate('full128',0,128,[np.eye(128)]*len(targets))
  for seed in range(3):
   maps,sv,sr,M,pa=subspaces(X[0],S[0],ev,seed);audits.append(dict(audit,seed=seed,pair_audit=pa));U=maps['predictive']
   np.savez_compressed(O/f'{ds}_basis_s{seed}.npz',**maps,singular_values=sv,residual_singular_values=sr,cross_matrix=M)
   cross=(X[0]-X[0].mean(0)).T@((Y[0]-Y[0].mean(0))/Y[0].std(0))/len(X[0]);ee=(U.T@cross)**2;frac=np.cumsum(ee,axis=0)/ee.sum(0)
   for j,t in enumerate(targets):energy.append({'dataset':ds,'seed':seed,'target':t,'spectral_train_covariance_energy':ee[:,j].tolist(),'top4_fraction':float(frac[3,j]),'top8_fraction':float(frac[7,j]),'top1_fraction':float(frac[0,j]),'rank_for50pct':int(np.searchsorted(frac[:,j],.5)+1),'rank_for80pct':int(np.searchsorted(frac[:,j],.8)+1),'max_energy_axis_1based':int(ee[:,j].argmax()+1),'top4_prediction_energy_fraction':float(np.sum(sv[:4]**2)/np.sum(sv**2)),'top8_prediction_energy_fraction':float(np.sum(sv[:8]**2)/np.sum(sv**2))})
   # Per-target fractions are scale invariant, but components depend on the chosen Helmert basis.
   diffids=[j for j,t in enumerate(targets) if t.startswith('helmert_difference')]
   DD=Y[0][:,diffids]-Y[0][:,diffids].mean(0);cc=(X[0]-X[0].mean(0)).T@DD/len(X[0]);raw=(U.T@cc)**2;joint=raw.sum(1);joint/=joint.sum()
   rot,_=la.qr(np.random.default_rng(401).normal(size=(len(diffids),len(diffids))));rotated=(U.T@(cc@rot))**2;rjoint=rotated.sum(1);rjoint/=rjoint.sum()
   scaled=cc*np.arange(1,len(diffids)+1)[None,:]*3.7;sc=(U.T@scaled)**2;per=raw/raw.sum(0);sper=sc/sc.sum(0)
   association.append({'dataset':ds,'seed':seed,'joint_difference_top4_fraction':float(joint[:4].sum()),'joint_difference_top8_fraction':float(joint[:8].sum()),'joint_difference_rank_for80pct':int(np.searchsorted(np.cumsum(joint),.8)+1),'joint_orthogonal_contrast_rotation_error':float(np.max(np.abs(joint-rjoint))),'per_target_scale_invariance_error':float(np.max(np.abs(per-sper))),'definition':'Joint contrast energy uses raw Helmert contrasts of train-standardized original factors; no per-contrast variance rescaling before summation. Frobenius energy is invariant to orthogonal contrast rotation. Individual target fractions remain invariant to nonzero scale; per-contrast values depend on chosen Helmert basis.'})
   assert association[-1]['joint_orthogonal_contrast_rotation_error']<1e-10 and association[-1]['per_target_scale_invariance_error']<1e-10
   for k in [4,8]:
    for name,Q in maps.items():
     B=Q[:,:k];assert np.max(np.abs(B.T@B-np.eye(k)))<1e-9
     evaluate(name,seed,k,[B]*len(targets),future=True)
     if seed==0:base[(name,k)]=B
     else:
      cs=la.svdvals(base[(name,k)].T@B);stability.append({'dataset':ds,'method':name,'seed':seed,'rank':k,'mean_squared_principal_cosine':float(np.mean(cs**2)),'minimum_principal_cosine':float(cs.min())})
    for name,Q in [('supervised_best_spectral',U)]+([('supervised_best_original',q)] if seed==0 else []):
     ax=X[0]@Q;cs=(ax-ax.mean(0)).T@((Y[0]-Y[0].mean(0))/Y[0].std(0))/len(ax);gs=(cs**2).reshape(128//k,k,-1).sum(1);chosen=gs.argmax(0);Bs=[Q[:,k*g:k*(g+1)] for g in chosen];evaluate(name,seed,k,Bs,chosen)
   print(ds,'seed',seed,'complete',flush=True)
  write(O/'results.json',rows);write(O/'future_results.json',frows);write(O/'audits.json',audits);write(O/'stability.json',stability);write(O/'spectral_energy.json',energy);write(O/'association_invariance.json',association)
 write(O/'COMPUTATION_COMPLETE.json',{'factor_rows':len(rows),'future_rows':len(frows),'status':'computed; independent validation pending'})

def verify(root,O):
 rows=json.loads((O/'results.json').read_text());pmax=0.;rmax=0.;fmax=0.;count=0;nfuture=0;decomp=0.;rotation_error=0.;cross_error=0.;spectrum_gaps=[]
 for ds in ['pong','pinball']:
  X,S,Y,targets,_=load(root,ds)
  for seed in range(3):
   z=np.load(O/f'{ds}_basis_s{seed}.npz');U=z['predictive'];M=z['cross_matrix'];sv=z['singular_values'];err=np.max(np.abs((M@M.T)@U-U*(sv**2)));decomp=max(decomp,float(err));assert err<1e-9
   # Independently reconstruct temporal cross moments and verify orthogonal-coordinate equivariance.
   ww=weights(len(X[0]),seed);mu=np.sum(X[0]*ww[:,None],axis=0)/ww.sum();xx=X[0]-mu;blocks=[]
   for h in LAGS:
    pw=np.minimum(ww[:-h],ww[h:]);blocks.append(np.einsum('ni,nj,n->ij',xx[:-h],xx[h:],pw)/pw.sum())
   rebuilt=np.hstack(blocks)/2;cross_error=max(cross_error,float(np.max(np.abs(rebuilt-M))))
   Q,_=np.linalg.qr(np.random.default_rng(810).normal(size=(128,128)));rot=np.hstack([Q.T@C@Q for C in blocks])/2;uu,_,_=np.linalg.svd(rot,full_matrices=False)
   for k in [4,8]:
    BB=Q@uu[:,:k];rotation_error=max(rotation_error,float(np.max(np.abs(BB@BB.T-U[:,:k]@U[:,:k].T))));spectrum_gaps.append({'dataset':ds,'seed':seed,'rank':k,'relative_boundary_squared_singular_gap':float((sv[k-1]**2-sv[k]**2)/sv[k-1]**2)})
   assert cross_error<1e-9 and rotation_error<1e-8
  for f in O.glob(ds+'_*.npz'):
   if '_basis_' in f.name:continue
   z=np.load(f);assert np.array_equal(z['source_dev'],S[2]) and np.array_equal(z['truth'],Y[2]);p=z['prediction'];maps=z['maps'];k=maps.shape[-1]
   for j,t in enumerate(targets):
    B=maps[j];a=X[0]@B;mx=a.mean(0);my=Y[0][:,j].mean();xc=a-mx;rhs=xc.T@(Y[0][:,j]-my);G=xc.T@xc
    ws=[np.linalg.solve(G+len(xc)*al*np.eye(k),rhs) for al in ALPHAS];ls=[np.mean(((X[1]@B-mx)@w+my-Y[1][:,j])**2) for w in ws];b=int(np.argmin(ls));assert ALPHAS[b]==z['alpha'][j];pp=(X[2]@B-mx)@ws[b]+my;pmax=max(pmax,float(np.max(np.abs(pp-p[:,j]))));count+=1
    # Parse exact originating metrics from filename.
    stem=f.stem[len(ds)+1:];method,rest=stem.rsplit('_s',1);seed=int(rest.split('_')[0]);row=next(r for r in rows if r['dataset']==ds and r['method']==method and r['seed']==seed and r['rank']==k and r['target']==t);rmax=max(rmax,abs(float(r2(Y[2][:,j:j+1],pp[:,None])[0])-row['dev_r2']))
   if 'future_coefficients' in z:
    B=maps[0];a=X[0][:-8]@B;mx=a.mean(0);yy=np.concatenate([X[0][h:len(X[0])-8+h] for h in LAGS],axis=1)/2;my=yy.mean(0);xc=a-mx;w=np.linalg.solve(xc.T@xc+len(xc)*float(z['future_alpha'])*np.eye(k),xc.T@(yy-my));yt=np.concatenate([X[2][h:len(X[2])-8+h] for h in LAGS],axis=1)/2;pp=(X[2][:-8]@B-mx)@w+my;se=np.mean((pp-yt)**2,axis=1);fmax=max(fmax,float(np.max(np.abs(se-z['future_dev_mse_per_row']))));assert np.array_equal(z['future_source'],S[2][:-8]);nfuture+=1
 assert pmax<1e-8 and rmax<1e-9 and fmax<1e-9
 v={'status':'passed','factor_prediction_arrays':count,'future_factored_predictors':nfuture,'prediction_max_abs_error':pmax,'r2_max_abs_error':rmax,'future_row_mse_max_abs_error':fmax,'svd_eigen_equation_max_error':decomp,'cross_matrix_reconstruction_max_error':cross_error,'rotation_equivariance_projector_max_error':rotation_error,'spectrum_boundary_gaps':spectrum_gaps,'source_and_targets':'exact match original cache','script_sha256':sha(Path(__file__))};write(O/'independent_validation.json',v);write(O/'manifest.json',{p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['manifest.json','COMPLETE.json']});write(O/'COMPLETE.json',v);print(json.dumps(v),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--verify',action='store_true');a=ap.parse_args();(verify if a.verify else run)(a.root,a.output)
