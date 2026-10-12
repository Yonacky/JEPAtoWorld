"""Post-hoc matched-256 capacity diagnostic; not a new confirmation."""
from pathlib import Path
import argparse,sys,time,numpy as np,torch
from complex_haar_evaluate import fit_readout,H
from complex_spatial_evaluate import write,sha
p=argparse.ArgumentParser()
for k in ['root','input','previous','output']:p.add_argument('--'+k,type=Path,required=True)
p.add_argument('--device',default='auto');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);assert not any(a.output.iterdir())
sys.path.insert(0,str(a.root/'core/e03'))
from device_utils import activate_device
device=activate_device(a.device);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
start=time.time()
write(a.output/'config.json',{'scope':'Post-hoc after observing fixed64+64 primary failure; same already inspected holdout; exploratory only','rank':256,'models':['mean256','grid256','C128_D128','C128_D128_group','C128_D128_joint_white'],'source_sha256':sha(Path(__file__)),'helper_sha256':sha(Path(__file__).with_name('complex_haar_evaluate.py')),'features_sha256':sha(a.input/'features.npz'),'previous_results_sha256':sha(a.previous/'results.json')})
m=dict(np.load(a.input/'sample_manifest.npz',allow_pickle=False));f=dict(np.load(a.input/'features.npz',allow_pickle=False));maps=dict(np.load(a.previous/'transforms.npz',allow_pickle=False))
mask=[m['split']==i for i in range(3)];Y=[m['labels'][q].astype(float) for q in mask];std=Y[0].std(0)
raw=f['grid'].astype(float).reshape(-1,1280,4)@H.T
C=raw[:,:,0];D=raw[:,:,1:].reshape(-1,3840)
c=(C-maps['C_mean'])@maps['C_basis128'];d=(D-maps['D_mean'])@maps['D_basis128']
X={'C128_D128':[np.concatenate([c[q],d[q]],1) for q in mask]};newmaps={}
for kind in ['mean','grid']:
 t=torch.as_tensor(f[kind],device=device,dtype=torch.float64);tr=t[torch.as_tensor(mask[0],device=device)];mu=tr.mean(0)
 _,s,v=torch.linalg.svd(tr-mu,full_matrices=False);assert s[255]>s[0]*1e-8
 b=v[:256].T/(s[:256]/np.sqrt(len(tr)-1))
 X[kind+'256']=[((t[torch.as_tensor(q,device=device)]-mu)@b).cpu().numpy() for q in mask]
 newmaps[kind+'_mean']=mu.cpu().numpy();newmaps[kind+'_basis256']=b.cpu().numpy()
j=X['C128_D128'];mu=j[0].mean(0);t=torch.as_tensor(j[0]-mu,device=device,dtype=torch.float64);_,s,v=torch.linalg.svd(t,full_matrices=False);keep=s>s[0]*1e-10
b=(v[keep].T/(s[keep]/np.sqrt(len(t)-1))).cpu().numpy();X['C128_D128_joint_white']=[(q-mu)@b for q in j];newmaps['joint_mean']=mu;newmaps['joint_basis']=b
X['C128_D128_group']=j
rows=[];pred={}
for key,x in X.items():
 kw={'common':128,'ratios':(0.,.1,.3,1.,3.,10.)} if key.endswith('_group') else {}
 pr,fit=fit_readout(x,Y,device,**kw);np.savez_compressed(a.output/(key+'_fit.npz'),**fit);pred[key]=pr
 err=((pr-Y[2])/std)**2;eps=m['episode'][mask[2]];per=np.asarray([err[eps==ep].mean(0) for ep in np.unique(eps)])
 r2=1-((pr-Y[2])**2).sum(0)/((Y[2]-Y[2].mean(0))**2).sum(0)
 rows.append({'key':key,'r2':r2.tolist(),'nmse':per.mean(0).tolist(),'per_sequence_nmse':per.tolist(),'ratio_alpha':fit['ratio_alpha_candidates'][fit['selected_candidate']].tolist()})
 print(key,flush=True)
np.savez_compressed(a.output/'predictions.npz',**pred);np.savez_compressed(a.output/'transforms.npz',**newmaps)
write(a.output/'results.json',{'scope':'post-hoc capacity diagnostic only','targets':m['target_names'].tolist(),'rows':rows})
write(a.output/'COMPLETE.json',{'complete':True,'seconds':time.time()-start,'source_sha256':sha(Path(__file__))})
