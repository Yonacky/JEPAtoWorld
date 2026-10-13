import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import numpy as np,json,hashlib
R=Path('runs/intervention_20261012');O=Path('runs/nonorthogonal_edit_20261013');m=np.load(O/'models.npz');U=m['U'];initial=np.load(R/'step2b/fitted.npz')['joint_0'];result=json.loads((O/'results.json').read_text());out={'status':'PASS','domains':{},'paired_bootstrap':{}};paths=[]
for p,h in json.loads((O/'config.json').read_text())['input_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
np.testing.assert_allclose(U.T@U,np.eye(64),atol=1e-10)
for domain in ['train','tune','id','ood']:
 dp=R/('pilot/dataset.npz' if domain in ['train','tune'] else f'tworef_{domain}_data/dataset.npz');fp=R/('features/features.npz' if domain in ['train','tune'] else f'tworef_{domain}_features/features.npz');paths.extend([dp,fp]);z=np.load(dp);x=np.load(fp)['grid'].astype(float);pairs=json.loads((O/(domain+'_pairs.json')).read_text());saved=np.load(O/(domain+'_prediction_samples.npz'));maxpred=maxmetric=0.;count=0
 for name in result[domain]['models']:
  rows=[];leaks=[]
  if name in m.files:W=m[name];wi=np.linalg.solve(W,np.eye(64));P=[W[:,k*16:(k+1)*16]@wi[k*16:(k+1)*16] for k in range(4)];assert all(np.linalg.matrix_rank(p,tol=1e-8)==16 for p in P);np.testing.assert_allclose(sum(P),np.eye(64),atol=1e-10)
  else:P=[np.zeros((64,64))]*3 if name=='no_edit' else initial
  for pair in pairs:
   sc=pair['recipient'];donor=pair['donor'];ix=np.flatnonzero(z['scene']==sc);iy=np.flatnonzero(z['scene']==donor);X=x[ix];Y=x[iy];c=z['codes'][ix];assert np.array_equal(c,z['codes'][iy]);assert len(c)==27;H=X@U;G=Y@U
   for k in range(3):
    triples=[(a,b,int(sum((c[b,j] if j==k else c[a,j])*[9,3,1][j] for j in range(3)))) for a in range(27) for b in range(27) if c[a,k]!=c[b,k] and any(c[a,j]!=c[b,j] for j in range(3) if j!=k)];a,b,t=np.array(triples).T;v=G[b]-H[a]
    if name!='joint0_uncalibrated':v-=((Y[[0,26]].mean(0)-X[[0,26]].mean(0))@U)
    if name in m.files:
     co=v@W;co[:,:16*k]=0;co[:,16*(k+1):]=0;pred=np.linalg.solve(W.T,co.T).T
    else:pred=v@P[k]
    maxpred=max(maxpred,float(np.max(abs(pred[:7]-saved[name+f'_s{sc}_f{k}']))));den=np.sum((X[t]-X[a])**2,1);edited=X[a]+pred@U.T;err=np.sum((edited-X[t])**2,1)/den
    # Direct full-space nearest search, batched across 27 candidate outcomes.
    distances=np.sum(edited*edited,1)[:,None]+np.sum(X*X,1)[None,:]-2*edited@X.T;direct=np.stack([np.sum((edited[:7]-X[j])**2,1) for j in range(27)],1);np.testing.assert_allclose(distances[:7],direct,atol=1e-7);nearest=distances.argmin(1);other=[j for j in range(3) if j!=k];rows.append([err.mean(),np.mean(np.sum(((H[t]-H[a])@P[k])**2,1)/den),np.mean(c[nearest,k]==c[b,k]),np.mean(np.all(c[nearest][:,other]==c[a][:,other],1))]);count+=len(a)
    ll=[]
    for j in other:
     ij=[(a,b) for a in range(27) for b in range(27) if c[a,j]!=c[b,j] and all(c[a,h]==c[b,h] for h in range(3) if h!=j)];aa,bb=np.array(ij).T;dd=X[bb]-X[aa];ll.append(np.mean(np.sum(((dd@U)@P[k])**2,1)/np.sum(dd*dd,1)))
    leaks.append([k,np.mean(ll)])
  arr=np.array(rows).reshape(-1,3,4);original=np.load(O/(domain+'_'+name+'_scene.npz'))['metrics'];maxmetric=max(maxmetric,float(np.max(abs(arr-original))));np.testing.assert_allclose(arr,original,atol=1e-9);leak=[np.mean([v for j,v in leaks if j==k]) for k in range(3)];np.testing.assert_allclose(leak,result[domain]['models'][name]['cross_factor_leakage'],atol=1e-10)
 out['domains'][domain]={'full_edits_checked':count,'prediction_maxabs':maxpred,'scene_metric_maxabs':maxmetric};print(domain,out['domains'][domain],flush=True)
 if domain in ['id','ood']:
  for ref in ['joint0_twoanchors','orth_lambda1']:
   a=np.load(O/(domain+'_oblique_lambda1_scene.npz'))['metrics'];b=np.load(O/(domain+'_'+ref+'_scene.npz'))['metrics'];d=a-b;half=len(d)//2;d=(d[:half]+d[half:])/2;rng=np.random.default_rng(20261013);boot=d[rng.integers(0,half,(5000,half))].mean(1);out['paired_bootstrap'][domain+'-minus-'+ref]={'factor_means':d.mean(0).tolist(),'factor_ci95':np.quantile(boot,[.025,.975],axis=0).tolist(),'overall_error_difference':float(d[:,:,0].mean()),'overall_error_ci95':np.quantile(boot[:,:,0].mean(1),[.025,.975]).tolist(),'independent_pairs':half}
sel=json.loads((O/'selection.json').read_text());assert sel=={f:min([f+'_lambda0',f+'_lambda1'],key=lambda n:result['tune']['models'][n]['selection_score']) for f in ['orth','oblique']};out['input_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))};out['bootstrap_scope']='5000 fixed-seed pair-cluster bootstrap; reused exploratory sets; no multiplicity correction';(O/'independent_validation.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
