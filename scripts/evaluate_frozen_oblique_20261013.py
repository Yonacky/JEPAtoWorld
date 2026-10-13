"""G7i fixed-model evaluation only. No fitting, selection, or label-dependent search."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='2'
import argparse,hashlib,json,itertools
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main(a):
 cfg=json.loads(a.frozen_config.read_text(encoding='utf-8'));O=a.output;assert not O.exists(),'preserve output';O.mkdir(parents=True)
 for key,p in [('models',a.models),('joint0',a.joint0),('selection',a.selection)]:assert sha(p)==cfg['frozen_hashes'][key],key
 z=np.load(a.data);f=np.load(a.features);meta=json.loads(a.features.with_name('features.json').read_text());manifest=np.load(a.features.with_name('sample_manifest.npz'))
 assert sha(a.features)==meta['features_sha256'];assert array_sha(z['images'])==meta['images_sha256'];assert list(z['images'].shape)==meta['images_shape']
 for k,v in cfg['encoder_contract'].items():assert meta[k]==v,('encoder contract',k)
 for k in manifest.files:
  if k in meta['manifest_hashes']:assert array_sha(manifest[k])==meta['manifest_hashes'][k]['sha256'],k
 np.testing.assert_array_equal(z['target_names'],cfg['target_names'])
 for k in ['labels','split','target_names']:np.testing.assert_array_equal(manifest[k],z[k])
 np.testing.assert_array_equal(manifest['episode'],z['scene']);np.testing.assert_array_equal(manifest['frame'],z['codes']@np.array([9,3,1]));assert np.all(manifest['episode_length']==27)
 x=f['grid'].astype(np.float64);m=np.load(a.models);U=m['U'];base=np.load(a.joint0)['joint_0'];assert len(x)==len(z['scene']) and x.shape[1]==len(U);assert np.isfinite(x).all();np.testing.assert_allclose(U.T@U,np.eye(64),atol=1e-10)
 scenes=sorted(np.unique(z['scene']).tolist());assert len(scenes)%2==0
 if a.old_alignment:
  assert sha(a.data)==cfg['old_alignment_data_sha256'];half=len(scenes)//2;scenes=scenes[:2]+scenes[half:half+2]
 else:assert len(scenes)==64,'frozen confirmatory domain requires 64 scenes'
 names=cfg['models'];assert names==['no_edit','joint0_twoanchors','orth_lambda1','oblique_lambda1'];assert cfg['bootstrap_seed']==20261013 and cfg['bootstrap_draws']==10000 and cfg['expected_production_scenes']==64;Ps={'no_edit':np.zeros_like(base),'joint0_twoanchors':base}
 for name in ['orth_lambda1','oblique_lambda1']:
  W=m[name];Wi=np.linalg.solve(W,np.eye(64));Ps[name]=np.array([W[:,16*k:16*k+16]@Wi[16*k:16*k+16] for k in range(3)])
 tasksets=[(0,),(1,),(2,),(0,1),(0,2),(1,2)];tasknames=['position','color','background','position+color','position+background','color+background'];codes=np.array(list(itertools.product(range(3),repeat=3)));records={};metrics=np.empty((4,len(scenes),6,2,4));counts=np.empty((len(scenes),6,2),int);pairlist=[];maxdirect=0.;samples={};details={};half=len(scenes)//2
 # Fixed records; the same rows and masks serve all models.
 for si,sc in enumerate(scenes):
  donor=scenes[(si+half)%len(scenes)];ids=np.flatnonzero(z['scene']==sc);di=np.flatnonzero(z['scene']==donor);assert len(ids)==27 and len(di)==27;np.testing.assert_array_equal(z['codes'][ids],codes);np.testing.assert_array_equal(z['codes'][di],codes);X=x[ids];Y=x[di];H=X@U;G=Y@U;offset=G[[0,26]].mean(0)-H[[0,26]].mean(0);norm=np.sum(X*X,1);distance=norm[:,None]+norm[None,:]-2*X@X.T;pairlist.append({'recipient':sc,'donor':donor,'recipient_rows':ids.tolist(),'donor_rows':di.tolist()})
  for ti,ks in enumerate(tasksets):
   other=[k for k in range(3) if k not in ks];aidx,didx=np.where(np.all(codes[:,None,list(ks)]!=codes[None,:,list(ks)],axis=2)&np.any(codes[:,None,other]!=codes[None,:,other],axis=2));tc=codes[aidx].copy();tc[:,list(ks)]=codes[didx][:,list(ks)];target=tc@np.array([9,3,1]);mask=~(np.isin(aidx,[0,26])|np.isin(didx,[0,26])|np.isin(target,[0,26]));assert mask.any();den=np.sum((X[target]-X[aidx])**2,1);assert np.all(den>0);t=H[target]-H[aidx];v=G[didx]-H[aidx]-offset;records[f's{sc}_t{ti}']=np.stack([aidx,didx,target,mask.astype(int)],1);counts[si,ti]=[len(mask),mask.sum()]
   for ni,name in enumerate(names):
    P=sum(Ps[name][k] for k in ks);pred=v@P;err=1+(np.sum(pred*pred,1)-2*np.sum(pred*t,1))/den;dist=distance[aidx]+2*np.einsum('ni,nji->nj',pred,H[aidx,None]-H[None])+np.sum(pred*pred,1)[:,None];near=dist.argmin(1);targetok=np.all(codes[near][:,list(ks)]==codes[didx][:,list(ks)],1);keep=np.all(codes[near][:,other]==codes[aidx][:,other],1);exact=near==target;vals=np.stack([err,targetok,keep,exact],1);metrics[ni,si,ti,0]=vals.mean(0);metrics[ni,si,ti,1]=vals[mask].mean(0);details[f'{name}_s{sc}_t{ti}']=np.column_stack([vals,near]);samples[f'{name}_s{sc}_t{ti}']=pred[:7]
    edited=X[aidx[:7]]+pred[:7]@U.T;direct=np.sum((edited-X[target[:7]])**2,1)/den[:7];maxdirect=max(maxdirect,float(np.max(abs(direct-err[:7]))));directdist=np.stack([np.sum((edited-X[j])**2,1) for j in range(27)],1);np.testing.assert_array_equal(directdist.argmin(1),near[:7])
  print('scene',sc,'complete',flush=True)
 assert maxdirect<1e-8
 np.savez_compressed(O/'scene_metrics.npz',metrics=metrics,counts=counts,scene=np.array(scenes));np.savez_compressed(O/'edit_records.npz',**records);np.savez_compressed(O/'edit_metrics.npz',**details);np.savez_compressed(O/'prediction_samples.npz',**samples);dump(O/'pair_map.json',pairlist)
 # Equal factor weighting within scene, then bidirectional scene-pair mean.
 pairmetrics=(metrics[:,:half]+metrics[:,half:])/2;rng=np.random.default_rng(cfg['bootstrap_seed']);ix=rng.integers(0,half,(10000,half));intervals={}
 for ref in ['orth_lambda1','joint0_twoanchors']:
  delta=pairmetrics[names.index('oblique_lambda1')]-pairmetrics[names.index(ref)];boot=delta[ix].mean(1);primary=np.stack([delta[:,:3,1,0].mean(1),delta[:,1,1,1]],1);bp=primary[ix].mean(1);intervals[ref]={'metric_difference_mean':delta.mean(0).tolist(),'metric_ci95':np.quantile(boot,[.025,.975],axis=0).tolist(),'primary_names':['single_factor_macro_relative_error_anchor_excluded','single_color_target_recovery_anchor_excluded'],'primary_difference':primary.mean(0).tolist(),'primary_ci95':np.quantile(bp,[.025,.975],axis=0).tolist(),'primary_ci98_75':np.quantile(bp,[.00625,.99375],axis=0).tolist(),'primary_status':'predeclared family of four only for orth_lambda1 across ID/OOD; joint0 is secondary'}
 result={'domain':a.domain,'scope':'old-data alignment only' if a.old_alignment else 'fixed candidates on new domain; no fit or selection','models':names,'tasks':tasknames,'mask_order':['all_edits_sensitivity','anchor_excluded_primary'],'metric_order':['full_space_relative_error','all_target_factors_correct','all_nontarget_factors_preserved','exact_target_state_retrieval'],'scene_means':metrics.mean(1).tolist(),'contrasts':intervals,'scene_count':len(scenes),'independent_pairs':half,'bootstrap_draws':10000,'direct_full_error_maxabs':maxdirect,'denominator':'full-feature squared source-to-real-target distance','retrieval_candidates':'all 27 recipient states including anchors'}
 if a.old_alignment:
  old=json.loads(a.old_results.read_text());oldm=[]
  for ni,n in enumerate(names):
   arr=np.load(a.old_results.parent/('id_'+n+'_scene.npz'))['metrics'];idx=[0,1,16,17];np.testing.assert_allclose(metrics[ni,:,0:3,0,0],arr[idx,:,0],atol=1e-10);np.testing.assert_allclose(metrics[ni,:,0:3,0,1:3],arr[idx,:,2:4],atol=1e-10);oldm.append(float(np.max(abs(metrics[ni,:,0:3,0,0]-arr[idx,:,0]))))
  result['old_alignment_maxabs']=oldm
 for key,p in [('models',a.models),('joint0',a.joint0),('selection',a.selection)]:assert sha(p)==cfg['frozen_hashes'][key],('post-evaluation hash',key)
 dump(O/'results.json',result);sources=[a.data,a.features,a.features.with_name('features.json'),a.features.with_name('sample_manifest.npz'),a.models,a.joint0,a.selection,a.frozen_config,Path(__file__)];dump(O/'run_config.json',{'arguments':{k:str(v) for k,v in vars(a).items()},'frozen_config':cfg,'source_sha256':{str(p):sha(p) for p in sources}});dump(O/'manifest.json',{'files':[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(O.iterdir()) if p.is_file()]});print('COMPLETE',maxdirect,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--joint0',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--frozen-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--domain',choices=['id','ood'],required=True);p.add_argument('--old-alignment',action='store_true');p.add_argument('--old-results',type=Path,default=Path('runs/nonorthogonal_edit_20261013/results.json'));main(p.parse_args())
