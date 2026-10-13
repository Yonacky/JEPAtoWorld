"""G7c encoder-family frozen scalar readouts; fork of G5j."""

import os

for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'

from pathlib import Path

import argparse,json,hashlib

import numpy as np

from scipy import linalg as la

ALPHAS=[1e-5,.001,.1,1.,10.]

def sha(p):

 h=hashlib.sha256()

 with p.open('rb') as f:

  for b in iter(lambda:f.read(2**20),b''):h.update(b)

 return h.hexdigest()

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def metrics(y,p,sc,sd):

 e=(y-p)**2;scene={str(s):float(e[sc==s].mean()) for s in np.unique(sc)};return {'log_mse':float(np.mean(list(scene.values()))),'train_standardized_mse':float(np.mean(list(scene.values()))/sd**2),'r2':float(1-e.sum()/np.sum((y-y.mean())**2)),'scene_log_mse':scene}

def main():

 ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--features',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--input-kind',choices=['image_g5j','branch_vector'],default='image_g5j');ap.add_argument('--dataset-manifest',type=Path);ap.add_argument('--window-spec',type=Path);a=ap.parse_args();O=a.output;O.mkdir(parents=True,exist_ok=False)

 z=np.load(a.data);fz=np.load(a.features);manifest_path=a.dataset_manifest or a.features.parent/'sample_manifest.npz';fm=np.load(manifest_path);grid=fz['grid' if a.input_kind=='image_g5j' else 'branch_vector'].astype(float);assert grid.ndim==2 and grid.shape[1]>0 and np.isfinite(grid).all()

 if a.input_kind=='image_g5j':

  assert grid.shape[0]==6480 and len(z['scene'])==6480

 else:

  assert grid.shape[0]==720 and a.dataset_manifest is not None and a.window_spec is not None

  window=json.loads(a.window_spec.read_text());assert window.get('task')=='retrospective_readout' and window.get('sample_steps') and window.get('pooling') and window.get('encoder_id');assert len(set(window['sample_steps']))==len(window['sample_steps']) and min(window['sample_steps'])>=0 and max(window['sample_steps'])<=120

  steps=np.asarray(window['sample_steps']);assert steps.ndim==1 and len(steps)>1 and np.all(steps==steps.astype(int)) and np.all(np.diff(steps)>0)
  assert len(z['scene'])==720*len(steps)
  for k in ['scene','contact_mode','mass_level','force_level','split','episode','labels','source_rows','clip_frames']:assert k in fm,k
  for k in ['scene','contact_mode','mass_level','force_level','split','episode']:assert fm[k].shape==(720,)
  source_rows=fm['source_rows'];assert source_rows.shape==(720,len(steps)) and np.issubdtype(source_rows.dtype,np.integer)
  assert np.array_equal(np.sort(source_rows.ravel()),np.arange(len(z['scene']))),'source rows must partition entire uniform dataset'
  np.testing.assert_array_equal(fm['clip_frames'],np.broadcast_to(steps,source_rows.shape))
  np.testing.assert_array_equal(z['frame'][source_rows],fm['clip_frames'])
  for k in ['scene','contact_mode','mass_level','force_level','split','episode']:
   np.testing.assert_array_equal(z[k][source_rows],np.broadcast_to(fm[k][:,None],source_rows.shape))
  np.testing.assert_array_equal(z['labels'][source_rows],np.broadcast_to(fm['labels'][:,None,:],(720,len(steps),3)))
  if 'time_seconds' in z:np.testing.assert_allclose(z['time_seconds'][source_rows],fm['clip_frames']*.004,rtol=0,atol=1e-12)

  branch_lookup={}

  for j in range(720):

   key=(int(fm['scene'][j]),str(fm['contact_mode'][j]),int(fm['mass_level'][j]),int(fm['force_level'][j]));assert key not in branch_lookup;branch_lookup[key]=j

 for k in ['split','episode','frame','labels']:

  if a.input_kind=='image_g5j':assert np.array_equal(z[k],fm[k]),k

 if a.input_kind=='image_g5j':

  for k in ['scene','contact_mode','mass_level','force_level']:

   if k in fm:assert np.array_equal(z[k],fm[k]),k

 md={k:z[k] for k in ['scene','contact_mode','split','episode','frame','mass_level','force_level','labels']};modecol=md['contact_mode'].astype(str);assert set(modecol)=={'on','off'}

 sources=[a.data,a.features,manifest_path]

 if a.window_spec:sources.append(a.window_spec)
 sources.extend(p for p in [a.data.parent/'config.json',a.data.parent/'scene_metadata.json',a.data.parent/'validation.json'] if p.exists())

 dump(O/'config.json',{'input_sha256':{str(p):sha(p) for p in sources},'script_sha256':sha(Path(__file__)),'alphas':ALPHAS,'rank':32,'split_scenes':[24,8,8],'targets':['logratio','logmass','logforce'],'input_kind':a.input_kind,'feature_dimension':int(grid.shape[1]),'window_spec':window if a.input_kind=='branch_vector' else {'sample_steps':[0,30,120],'operation':'concat(delta30,delta120)'},'input':'grid concat(delta30,delta120)' if a.input_kind=='image_g5j' else 'one branch vector, explicit window','target_scale':'train-only mean/std separately each mode/target','known_force':'logmass_hat=logFtrue-logratio_hat; additional true action, algebraically same ratio error, not independent evidence','selection':'tune scene-equal standardized MSE; five alpha ascending exact tie break','models':6})

 results=[];validation=[]

 for mode in ['on','off']:

  X=[];idx=[];sc=[];sp=[];Y=[]

  for scene in range(40):

   q=np.flatnonzero((md['scene']==scene)&(modecol==mode));assert len(q)==9*(9 if a.input_kind=='image_g5j' else len(steps))

   assert np.all(md['split'][q]==(0 if scene<24 else 1 if scene<32 else 2))

   for mi in range(3):

    for fi in range(3):

     ii=q[(md['mass_level'][q]==mi)&(md['force_level'][q]==fi)];ii=ii[np.argsort(md['frame'][ii])];assert np.array_equal(md['frame'][ii],[0,1,5,10,20,30,60,90,120] if a.input_kind=='image_g5j' else steps);assert len(np.unique(md['episode'][ii]))==1

     labels=md['labels'][ii];assert np.allclose(labels,[[.01,.02,.04][mi],.175,[.01,.02,.03][fi]],rtol=0,atol=1e-12)

     if a.input_kind=='image_g5j':

      v=grid[ii];X.append(np.r_[v[5]-v[0],v[8]-v[0]]);idx.append(ii)

     else:

      j=branch_lookup[scene,mode,mi,fi];np.testing.assert_array_equal(fm['source_rows'][j],ii);assert int(fm['split'][j])==int(md['split'][ii[0]]);X.append(grid[j]);idx.append([j])

      if 'labels' in fm:np.testing.assert_allclose(fm['labels'][j],labels[0],rtol=0,atol=1e-12)

      if 'episode' in fm:assert fm['episode'][j]==md['episode'][ii[0]]

     sc.append(scene);sp.append(int(md['split'][ii[0]]));m,f=labels[0,[0,2]];Y.append([np.log(f/m),np.log(m),np.log(f)])

  X=np.array(X);idx=np.array(idx);sc=np.array(sc);sp=np.array(sp);Y=np.array(Y);tr=sp==0;tu=sp==1;de=sp==2;assert (tr.sum(),tu.sum(),de.sum())==(216,72,72)

  assert not set(sc[tr])&set(sc[tu]) and not set(sc[tr])&set(sc[de]) and not set(sc[tu])&set(sc[de])

  mu=X[tr].mean(0);xc=X[tr]-mu;u,s,vh=la.svd(xc,full_matrices=False);rank=min(32,int(np.sum(s>max(s[0]*1e-8,1e-12))));P=vh[:rank].T;Z=[(X[q]-mu)@P for q in [tr,tu,de]];g=Z[0].T@Z[0]/tr.sum();uu,ss,vv=la.svd(Z[0],full_matrices=False)

  np.savez_compressed(O/(mode+'_pca.npz'),dataset_source_indices=idx if a.input_kind=='image_g5j' else fm['source_rows'][idx[:,0]],mean=mu,basis=P,singular_values=s,source_indices=idx,scene=sc,split=sp)

  for ti,target in enumerate(['logratio','logmass','logforce']):

   y=Y[:,ti];ym=y[tr].mean();sd=y[tr].std();assert sd>0;yt=(y[tr]-ym)/sd;c=Z[0].T@yt/tr.sum();ws=[la.solve(g+alpha*np.eye(rank),c,assume_a='pos') for alpha in ALPHAS];loss=np.array([np.mean([np.mean(((Z[1]@w-(y[tu]-ym)/sd)[sc[tu]==s])**2) for s in np.unique(sc[tu])]) for w in ws]);best=int(loss.argmin());pred=(Z[2]@ws[best])*sd+ym

   rhs=uu.T@yt;iws=[vv.T@(ss/(ss*ss+tr.sum()*alpha)*rhs) for alpha in ALPHAS];il=np.array([np.mean((Z[1]@w-(y[tu]-ym)/sd)**2) for w in iws]);assert int(il.argmin())==best;ip=(Z[2]@iws[best])*sd+ym;err=float(np.max(np.abs(ip-pred)));assert err<1e-7

   key=mode+'_'+target;np.savez_compressed(O/(key+'.npz'),truth=y[de],prediction=pred,scene=sc[de],coefficients=ws[best],alpha=ALPHAS[best],target_mean=ym,target_std=sd,tune_losses=loss,source_indices=idx[de],train_source_indices=idx[tr],tune_source_indices=idx[tu]);res={'mode':mode,'target':target,'rank':rank,'alpha':ALPHAS[best],'all_tune_losses':loss.tolist(),**metrics(y[de],pred,sc[de],sd)}

   pack=np.load(O/(key+'.npz'));direct=float(np.sum((pack['truth']-pack['prediction'])**2)/len(pack['truth']));assert abs(direct-res['log_mse'])<1e-12

   if ti==0:

    known=Y[de,2]-pred;knownmetrics=metrics(Y[de,1],known,sc[de],Y[tr,1].std());assert abs(knownmetrics['log_mse']-res['log_mse'])<1e-12;np.savez_compressed(O/(mode+'_knownforce_mass.npz'),truth=Y[de,1],prediction=known,true_logforce=Y[de,2],scene=sc[de],ratio_prediction=pred);res['known_force_mass']=knownmetrics

   results.append(res);validation.append({'model':key,'svd_prediction_max_error':err,'same_selected_alpha':True,'saved_prediction_mse_error':abs(direct-res['log_mse'])});print(key,res['log_mse'],res['r2'],flush=True)

 dump(O/'results.json',results);dump(O/'independent_validation.json',{'status':'passed','models':len(results),'checks':validation,'independence':'alternate SVD ridge solve and saved-prediction metric reduction; same data/PCA by design'})

 dump(O/'manifest.json',{str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in list(O.iterdir())+[Path(__file__)] if p.is_file() and p.name!='manifest.json'})

if __name__=='__main__':main()

