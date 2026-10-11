"""Explicit row-vector convention: future = current @ operator + intercept."""
import numpy as np
import scipy.linalg as la

ALPHAS=np.array([1e-4,1e-3,1e-2,.1,1.,10.,100.])
SCHEMA={
 'pong':{'ball_vel_dir':'angle','ball_vel_magn':'constant2','ball_x':'continuous','ball_y':'continuous','paddle_left_y':'continuous','paddle_right_y':'continuous','score_left':5,'score_right':5},
 'pinball':{'ball_x':'continuous','ball_x_vel':'continuous','ball_y':'continuous','ball_y_vel':'continuous','cyl_0_active':'continuous','cyl_1_active':'continuous','cyl_2_active':'continuous','cyl_3_active':'continuous','cyl_4_active':'continuous','paddle_left_y_pos':'continuous','paddle_right_y_pos':'continuous','score':20}}

def encode_targets(ds,latents,keys):
 assert set(keys)==set(SCHEMA[ds]),'unknown/missing factor key'
 assert latents.shape[1]==len(keys) and np.isfinite(latents).all()
 pieces=[];groups=[];j=0
 for k in keys:
  v=latents[:,keys.index(k)];typ=SCHEMA[ds][k]
  if typ=='constant2':assert np.allclose(v,2);continue
  if isinstance(typ,int):
   assert np.allclose(v,np.round(v),atol=1e-6),'noninteger categorical value'
   assert v.min()>=0 and v.max()<typ,'class out of range'
   y=np.eye(typ)[np.round(v).astype(int)];kind='categorical'
  elif typ=='angle':
   assert v.min()>=-1e-5 and v.max()<=2*np.pi+1e-5
   y=np.stack([np.sin(v),np.cos(v)],axis=1);kind='angle'
  else:
   if '_active' in k:assert v.min()>=-1e-6 and v.max()<=1+1e-6
   y=v[:,None];kind='continuous'
  pieces.append(y);groups.append({'name':k,'type':kind,'slice':[j,j+y.shape[1]]});j+=y.shape[1]
 return np.concatenate(pieces,axis=1),groups

def pair_indices(global_indices,horizon):
 assert horizon>0 and len(global_indices)>horizon
 before=global_indices[:-horizon];after=global_indices[horizon:]
 assert np.all(after-before==horizon),'noncontiguous/cross-segment pair'
 return before,after

def groups_fixed(n,b):
 assert n%b==0
 return [np.arange(i,i+b) for i in range(0,n,b)]

def mask_for(groups,n):
 assert sorted(np.concatenate(groups).tolist())==list(range(n)),'groups must partition coordinates'
 mask=np.ones((n,n))
 for g in groups:mask[np.ix_(g,g)]=0
 return mask

def ridge_fit(x,y,alpha):
 mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my
 w=la.solve(xc.T@xc/len(x)+alpha*np.eye(x.shape[1]),xc.T@yc/len(x),assume_a='pos')
 return w,mx,my

def forecast_fit(X,Y,groups,alphas=ALPHAS):
 # All means fitted on training pairs; one alpha chosen by tune MSE for all blocks.
 mx=X[0].mean(0);my=Y[0].mean(0);xx=[a-mx for a in X];yy=[a-my for a in Y]
 gram=xx[0].T@xx[0]/len(xx[0]);rhs=xx[0].T@yy[0]/len(xx[0]);best=np.inf
 for alpha in alphas:
  w=np.zeros_like(rhs)
  for g in groups:
   sl=np.ix_(g,g);w[sl]=la.solve(gram[sl]+alpha*np.eye(len(g)),rhs[sl],assume_a='pos')
  mse=np.mean((xx[1]@w-yy[1])**2)
  if mse<best:best=mse;wb=w;ab=float(alpha)
 return xx[2]@wb+my,{'weight':wb,'input_mean':mx,'output_mean':my,'alpha':ab,'tune_mse':float(best),'parameters':sum(len(g)**2 for g in groups)+len(my)}

def pca_whiten(x,k):
 x=np.asarray(x,dtype=np.float64);center=x.mean(0);xc=x-center
 assert k<=min(x.shape),'requested rank unavailable'
 # Exact covariance eigendecomposition; no randomized low-rank approximation.
 ev,V=la.eigh(xc.T@xc/len(x));order=np.argsort(ev)[::-1][:k];ev=ev[order];V=V[:,order]
 assert ev[-1]>1e-10*ev[0],'near-zero selected eigenvalue'
 w=V/np.sqrt(ev);z=xc@w
 assert np.max(np.abs(z.T@z/len(z)-np.eye(k)))<1e-7
 return center,w,ev

def lag_cov(x,lags):
 return np.stack([x[:-l].T@x[l:]/(len(x)-l) for l in lags])

def schur_basis(a):
 t,q=la.schur(a,output='real');assert np.max(np.abs(a-q@t@q.T))<1e-8
 groups=[];i=0
 while i<len(t):
  size=2 if i+1<len(t) and abs(t[i+1,i])>1e-9 else 1
  groups.append(np.arange(i,i+size));i+=size
 return q,t,groups

def optimize_joint(c,block,initializers,steps=250,device='cuda:0'):
 import torch
 n=c.shape[1];cc=torch.tensor(c,device=device,dtype=torch.float64);mask=torch.tensor(mask_for(groups_fixed(n,block),n),device=device);den=(cc*cc).sum().clamp_min(1e-15)
 def value(q):return ((((q.T@cc@q)*mask)**2).sum()/den)
 candidates=[torch.tensor(a,device=device,dtype=torch.float64) for a in initializers];costs=[float(value(q)) for q in candidates];chosen=int(np.argmin(costs));q=candidates[chosen];initial=costs[chosen];lr=1.;history=[initial]
 for it in range(steps):
  q.requires_grad_(True);f=value(q);g,=torch.autograd.grad(f,q);q=q.detach();a=g@q.T-q@g.T
  if float(torch.linalg.norm(a))<1e-8:break
  found=False
  for back in range(15):
   eye=torch.eye(n,device=device,dtype=torch.float64);qn=torch.linalg.solve(eye+lr*a/2,(eye-lr*a/2)@q);v=float(value(qn))
   if v<float(f)-1e-14:found=True;break
   lr*=.5
  if not found:break
  q=qn;history.append(v);lr=min(1.2*lr,20.)
 out=q.cpu().numpy();assert np.max(np.abs(out.T@out-np.eye(n)))<1e-8;assert np.max(np.diff(history),initial=0)<1e-12
 return out,{'initial_objective':initial,'final_objective':history[-1],'steps':len(history)-1,'initializer_index':chosen,'initializer_objectives':costs,'loss_history':history}

def factor_metrics(y,p,groups):
 result={}
 for g in groups:
  a,b=g['slice'];t=y[:,a:b];q=p[:,a:b]
  if g['type']=='categorical':
   tr=t.argmax(1);pr=q.argmax(1);v={'accuracy':float(np.mean(tr==pr)),'balanced_accuracy':float(np.mean([np.mean(pr[tr==k]==k) for k in np.unique(tr)]))}
  else:
   den=np.sum((t-t.mean(0))**2);v={'r2':float(1-np.sum((q-t)**2)/den) if den>1e-12 else None,'rmse':float(np.sqrt(np.mean((q-t)**2)))}
  result[g['name']]=v
 return result

def probe(X,Y,groups,budget):
 # Label-assisted diagnostic: select on train correlations only, tune alpha per factor.
 pred=np.empty_like(Y[2]);selected={};dense_cache=None
 if budget=='dense':
  mx=X[0].mean(0);xx=[v-mx for v in X];ev,V=la.eigh(xx[0].T@xx[0]/len(xx[0]));ev=np.maximum(ev,0)
  dense_cache=(mx,xx,ev,V)
 for g in groups:
  a,b=g['slice'];y=[v[:,a:b] for v in Y];ym=y[0].mean(0);ys=y[0].std(0);ys[ys<1e-8]=1;yz=[(v-ym)/ys for v in y]
  xm=X[0].mean(0);xs=X[0].std(0);xs[xs<1e-8]=1;corr=((X[0]-xm)/xs).T@yz[0]/len(X[0]);score=np.sum(corr*corr,axis=1)
  ind=np.arange(X[0].shape[1]) if budget=='dense' else np.argsort(score)[-int(budget):]
  selected[g['name']]=ind.tolist();xx=[v[:,ind] for v in X];best=np.inf
  if dense_cache is not None:mx,xc,ev,V=dense_cache
  else:
   mx=xx[0].mean(0);xc=[v-mx for v in xx];ev,V=la.eigh(xc[0].T@xc[0]/len(xc[0]));ev=np.maximum(ev,0)
  my=yz[0].mean(0);rhs=V.T@(xc[0].T@(yz[0]-my)/len(xc[0]))
  for alpha in ALPHAS:
   w=V@(rhs/(ev[:,None]+alpha));val=xc[1]@w+my;loss=np.mean((val-yz[1])**2)
   if loss<best:best=loss;pp=(xc[2]@w+my)*ys+ym
  pred[:,a:b]=pp
 return pred,selected
