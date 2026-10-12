"""Cached-feature rank audit only; no encoder or representation training."""
import argparse,json,pathlib,hashlib,sys
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--input',type=pathlib.Path,required=True);args=p.parse_args()
sys.path.insert(0,str(args.root/'core/e03'))
from device_utils import activate_device
device=activate_device('auto');torch.set_num_threads(4)
f=np.load(args.input/'features.npz')['mean'];m=np.load(args.input/'sample_manifest.npz')
X=[torch.as_tensor(f[m['split']==i],dtype=torch.float64,device=device) for i in range(3)]
Y=[torch.as_tensor(m['labels'][m['split']==i],dtype=torch.float64,device=device) for i in range(3)]
mu=X[0].mean(0);X=[x-mu for x in X];ym=Y[0].mean(0);yy=[y-ym for y in Y]
u,s,v=torch.linalg.svd(X[0],full_matrices=False);rank=int((s>s[0]*1e-10).sum());n=len(X[0])
A=[0.,1e-7,1e-5,1e-3,.01,.1,1.,10.,100.]
rows=[];predictions={};names=m['target_names'].tolist()
for k in sorted(set([32,64,128,256,rank])):
 if k>rank:continue
 for norm in ['white','unwhite']:
  z=[x@v[:k].T for x in X]
  if norm=='white':z=[a/(s[:k]/np.sqrt(n-1)) for a in z]
  # SVD makes train covariance diagonal; centering verified independently.
  gram=z[0].T@z[0]/n;diag=torch.diag(gram)
  error=float((gram-torch.diag(diag)).abs().max())
  assert error<1e-5*max(1.,float(diag.max())),error
  rhs=z[0].T@yy[0]/n
  fits=[];loss=[]
  for alpha in A:
   w=rhs/(diag[:,None]+alpha)
   fits.append(z[2]@w+ym);loss.append(((z[1]@w-yy[1])**2).mean(0))
  choice=torch.stack(loss).argmin(0)
  pred=torch.stack(fits)[choice,:,torch.arange(len(names),device=device)].T
  value=1-((Y[2]-pred)**2).sum(0)/((Y[2]-Y[2].mean(0))**2).sum(0)
  key=f'{norm}_{k}';predictions[key]=pred.cpu().numpy()
  for j,name in enumerate(names):rows.append({'rank':k,'normalization':norm,'target':name,'r2':float(value[j]),'alpha':A[int(choice[j])],'train_variance_retained':float(s[:k].square().sum()/s.square().sum())})
  print(norm,k,'done',flush=True)
report={'scope':'diagnostic rank sweep on pilot development data; not method selection or confirmation','rows':rows,'effective_training_rank':rank,'input_dimension':f.shape[1],'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'features_sha256':hashlib.sha256((args.input/'features.npz').read_bytes()).hexdigest(),'device':str(device),'full_unwhite':'full empirical training span is equivalent to original1280 linear ridge with isotropic coefficient norm; null train directions have zero coefficients','limitations':['Not a nonlinear upper bound','Different ranks have different readout capacity','Training-span rank is limited by sample count']}
(args.input/'rank_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False))
np.savez_compressed(args.input/'rank_predictions.npz',**predictions,target_dev=Y[2].cpu().numpy())
