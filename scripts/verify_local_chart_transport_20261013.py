"""Independent saved-data audit of G7h; alternate normal solves and ambient edits."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import pathlib,json,hashlib,numpy as np,itertools
ROOT=pathlib.Path('F:/2610/CVPR27');R=ROOT/'runs/intervention_20261012';O=ROOT/'runs/local_chart_transport_20261013'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=dict(np.load(O/'fitted.npz'));U=f['U'];P=f['P'];codes=np.array(list(itertools.product(range(3),repeat=3)));targets=[(0,),(1,),(2,),(0,1),(0,2),(1,2)]
meta=json.loads((O/'sample_edit_index.json').read_text());tr=np.load(O/'sample_edit_predictions.npz');scores={};maxdelta=maxrelative=maxnearest=maxfit=0.;count=0
for name,dat,feat,scenes in [('tune','pilot','features',list(range(32,40))),('id','tworef_id_data','tworef_id_features',list(range(32))),('ood','tworef_ood_data','tworef_ood_features',list(range(32)))]:
 z=np.load(R/dat/'dataset.npz');ss=z['scene'];cc=z['codes'];xx=np.load(R/feat/'features.npz')['grid'].astype(float);X=[]
 for sc in scenes:
  ids=np.flatnonzero(ss==sc);ids=ids[np.argsort(cc[ids]@np.array([9,3,1]))];X.append(xx[ids])
 X=np.array(X);H=X@U;anchors=(H[:,0]+H[:,26])/2;ch=dict(np.load(O/(name+'_charts.npz')));coords={};n=len(scenes)
 for method in ['global6','local6','wrong6']:
  b=ch[method];coords[method]=np.array([np.linalg.solve(t.T@t,t.T@(h-a).T).T for t,h,a in zip(b,H,anchors)])
 for key,domain,i,j,selection,method in meta:
  if domain!=name:continue
  p=tr[key];rec=p[:,0].astype(int);don=p[:,1].astype(int);tar=p[:,2].astype(int);ts=targets[selection]
  if method=='noedit':d=np.zeros((3,64))
  elif method in ('joint_raw','tworef'):
   off=anchors[j]-anchors[i] if method=='tworef' else 0;d=sum((H[j][don]-off-H[i][rec])@P[k] for k in ts)
  else:
   d=np.zeros((3,64))
   for k in ts:d+=(coords[method][j][don,2*k:2*k+2]-coords[method][i][rec,2*k:2*k+2])@ch[method][i,:,2*k:2*k+2].T
  maxdelta=max(maxdelta,float(np.abs(d-p[:,3:67]).max()));edit=X[i][rec]+d@U.T;dist=np.sum((edit[:,None]-X[i][None])**2,-1);den=np.sum((X[i][rec]-X[i][tar])**2,-1)
  maxrelative=max(maxrelative,float(np.abs(dist[np.arange(3),tar]/den-p[:,68]).max()));maxnearest=max(maxnearest,float(np.abs(dist.min(1)-p[:,67]).max()));count+=3
 # Full single-edit progress and displacement energy: err/noedit = 1-2*progress+edit_energy.
 out={}
 for method in ['tworef','global6','local6','wrong6']:
  vals=np.zeros((n,3,3))
  for i in range(n):
   j=(i+n//2)%n
   for k in range(3):
    other=[v for v in range(3) if v!=k];mask=(codes[:,None,k]!=codes[None,:,k])&np.any(codes[:,None,other]!=codes[None,:,other],axis=-1);rec,don=np.where(mask);tc=codes[rec].copy();tc[:,k]=codes[don,k];tar=tc@np.array([9,3,1]);true=X[i][tar]-X[i][rec];den=np.sum(true*true,1)
    if method=='tworef':d=(H[j][don]-anchors[j]+anchors[i]-H[i][rec])@P[k]
    else:d=(coords[method][j][don,2*k:2*k+2]-coords[method][i][rec,2*k:2*k+2])@ch[method][i,:,2*k:2*k+2].T
    prog=np.sum(d*(H[i][tar]-H[i][rec]),1)/den;en=np.sum(d*d,1)/den;vals[i,k]=[prog.mean(),en.mean(),(1-2*prog+en).mean()]
  out[method]=vals.mean(0).tolist()
 scores[name]=out;print(name,'sample check done',flush=True)
# Training fit, solve C normal equations instead of producer least-squares.
z=np.load(R/'pilot/dataset.npz');ss=z['scene'];xx=np.load(R/'features/features.npz')['grid'].astype(float);hs=np.array([xx[ss==i]@U for i in range(32)]);C=f['contrast_design']
for i,h in enumerate(hs):
 b=np.linalg.solve(C.T@C,C.T@(h-h[[0,26]].mean(0))).T;maxfit=max(maxfit,float(np.abs(b-f['Btrain'][i]).max()))
checks={'status':'PASS','sampled_edits':count,'delta_maxabs':maxdelta,'relative_error_maxabs':maxrelative,'nearest_distance_maxabs':maxnearest,'training_fit_maxabs':maxfit,'progress_order':['target_projection_progress','edit_energy_ratio','relative_error_recomputed'],'full_single_edit_progress':scores,'note':'fixed three examples per scene/selection/method; all single-edit progress computed; same data independent normal-equation inverses and explicit5120 reconstruction'}
assert max(maxdelta,maxrelative,maxnearest,maxfit)<1e-8
(O/'independent_validation.json').write_text(json.dumps(checks,indent=2));print({k:v for k,v in checks.items() if k!='full_single_edit_progress'},flush=True)
