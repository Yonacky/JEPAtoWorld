"""G7h fixed conditional six-coordinate chart transport. CPU only, old-dev exploration."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import argparse,json,hashlib,pathlib,itertools,time,shutil
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf8')
def load(root,dat,feat,scenes,U):
 with np.load(root/dat/'dataset.npz') as z:scene=z['scene'];codes=z['codes']
 with np.load(root/feat/'features.npz') as z:x=z['grid'].astype(np.float64)
 xs=[]
 for sc in scenes:
  ids=np.flatnonzero(scene==sc);order=np.argsort(codes[ids]@np.array([9,3,1]));ids=ids[order]
  np.testing.assert_array_equal(codes[ids],CODES);xs.append(x[ids])
 X=np.array(xs);return X,X@U
CODES=np.array(list(itertools.product(range(3),repeat=3)))
HELM=np.array([[1/np.sqrt(2),1/np.sqrt(6)],[-1/np.sqrt(2),1/np.sqrt(6)],[0,-2/np.sqrt(6)]])
Q=np.concatenate([HELM[CODES[:,j]] for j in range(3)],1);C=Q-Q[[0,26]].mean(0)
METHODS=['noedit','joint_raw','tworef','global6','local6','wrong6']

def describe(H):return np.concatenate([H[:,[0,26]].mean(1),H[:,26]-H[:,0]],1)
def chart_stats(B):
 sv=np.linalg.svd(B,compute_uv=False);rank=int((sv>sv[0]*1e-8).sum());A=np.linalg.pinv(B,rcond=1e-8)
 return A,{'singular_values':sv.tolist(),'effective_rank':rank,'condition':float(sv[0]/sv[-1]),'ill_conditioned':bool(sv[0]/sv[-1]>1e4),'inverse_norm':float(np.linalg.norm(A,2)),'forward_norm':float(sv[0]),'left_inverse_maxabs':float(np.abs(A@B-np.eye(6)).max())}
def anova(H):
 Z=H-H.mean(1,keepdims=True);main=np.zeros_like(Z);es=[]
 for k in range(3):
  mk=np.stack([Z[:,CODES[:,k]==l].mean(1) for l in range(3)],1)[:,CODES[:,k]];main+=mk;es.append(np.sum(mk*mk,(1,2)))
 total=np.sum(Z*Z,(1,2));inter=Z-main
 return {'centered_energy':total.tolist(),'main_effect_energy':np.array(es).T.tolist(),'interaction_energy':np.sum(inter*inter,(1,2)).tolist(),'interaction_fraction':(np.sum(inter*inter,(1,2))/total).tolist()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);a=ap.parse_args();r=a.root;o=a.output;o.mkdir(parents=True,exist_ok=False);t=time.time()
 with np.load(r/'step2b/fitted.npz') as f:U=f['U'];P=f['joint_0']
 np.testing.assert_allclose(U.T@U,np.eye(64),atol=1e-9,rtol=0)
 Xtr,Htr=load(r,'pilot','features',range(32),U);anchor=Htr[:,[0,26]].mean(1);desc=describe(Htr)
 Btrain=np.array([np.linalg.lstsq(C,h-aa,rcond=1e-8)[0].T for h,aa in zip(Htr,anchor)])
 D=np.sqrt(np.sum((desc[:,None]-desc[None])**2,-1));bandwidth=float(np.median(D[np.triu_indices(32,1)]));assert bandwidth>0
 fitted_res=Htr-anchor[:,None]-np.einsum('qj,sdj->sqd',C,Btrain)
 fit={'supervision':'train scene27 factor-level labels align Helmert chart; inference never sees factor codes','bandwidth':bandwidth,'pinv_rcond':1e-8,'condition_warning':1e4,'training_scenes':list(range(32)),'fit_residual_energy':np.sum(fitted_res**2,(1,2)).tolist(),'fit_residual_fraction':(np.sum(fitted_res**2,(1,2))/np.sum((Htr-Htr.mean(1,keepdims=True))**2,(1,2))).tolist(),'anova':anova(Htr),'train_chart_stats':[chart_stats(b)[1] for b in Btrain]}
 np.savez_compressed(o/'fitted.npz',U=U,P=P,Btrain=Btrain,descriptors=desc,bandwidth=bandwidth,contrast_design=C)
 dump(o/'fit.json',fit);allresults={};validation={'full_distance_maxabs':0.,'inverse_alternate_maxabs':0.,'cycle_maxabs':0.,'commutator_maxabs':0.};traces={};trace_index=[];sourcefiles=[r/'step2b/fitted.npz',r/'features/features.npz',r/'pilot/dataset.npz']
 for name,dat,feat,scenes in [('tune','pilot','features',list(range(32,40))),('id','tworef_id_data','tworef_id_features',list(range(32))),('ood','tworef_ood_data','tworef_ood_features',list(range(32)))]:
  X,H=load(r,dat,feat,scenes,U);n=len(scenes);aa=H[:,[0,26]].mean(1);dd=describe(H);dist2=np.sum((dd[:,None]-desc[None])**2,-1);logw=-dist2/(2*bandwidth**2);w=np.exp(logw-logw.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
  Bs={'global6':np.repeat(Btrain.mean(0)[None],n,axis=0),'local6':np.einsum('ns,sdj->ndj',w,Btrain),'wrong6':np.einsum('ns,sdj->ndj',w,np.roll(Btrain,1,axis=0))};As={};stats={};us={}
  for method,bs in Bs.items():
   pairs=[chart_stats(b) for b in bs];As[method]=np.array([v[0] for v in pairs]);stats[method]=[v[1] for v in pairs];us[method]=np.einsum('nqd,njd->nqj',H-aa[:,None],As[method])
   for i in range(n):
    alternate=np.linalg.lstsq(bs[i],(H[i]-aa[i]).T,rcond=1e-8)[0].T
    validation['inverse_alternate_maxabs']=max(validation['inverse_alternate_maxabs'],float(np.abs(alternate-us[method][i]).max()))
  chart_prediction_residual={k:np.sum((H-aa[:,None]-np.einsum('nqj,ndj->nqd',us[k],Bs[k]))**2,(1,2)).tolist() for k in Bs}
  np.savez_compressed(o/(name+'_charts.npz'),weights=w,scenes=scenes,anchors=aa,**{k:Bs[k] for k in Bs},**{k+'_inverse':As[k] for k in Bs},**{k+'_coordinates':us[k] for k in Bs})
  selections=[(k,) for k in range(3)]+list(itertools.combinations(range(3),2));results={m:np.zeros((n,6,5)) for m in METHODS};excluded={m:np.zeros((n,3,5)) for m in METHODS}
  for i,scene in enumerate(scenes):
   j=(i+n//2)%n;h,g=H[i],H[j];x=X[i];full_dist=np.sum((x[:,None]-x[None])**2,-1)
   for si,targets in enumerate(selections):
    other=[k for k in range(3) if k not in targets];mask=np.ones((27,27),bool)
    for k in targets:mask&=CODES[:,None,k]!=CODES[None,:,k]
    mask&=np.any(CODES[:,None,other]!=CODES[None,:,other],axis=-1)
    rec,don=np.where(mask);tc=CODES[rec].copy();tc[:,targets]=CODES[don][:,targets];tar=tc@np.array([9,3,1]);den=full_dist[rec,tar];assert den.min()>1e-8
    for method in METHODS:
     if method=='noedit':delta=np.zeros((len(rec),64))
     elif method in ('joint_raw','tworef'):
      offset=aa[j]-aa[i] if method=='tworef' else 0
      delta=(g[don]-h[rec]-offset)@P[list(targets)].sum(0)
     else:
      coord=np.zeros((len(rec),6))
      for k in targets:coord[:,2*k:2*k+2]=(us[method][j][don]-us[method][i][rec])[:,2*k:2*k+2]
      delta=coord@Bs[method][i].T
     ds=full_dist[rec]+2*np.einsum('nd,nqd->nq',delta,h[rec,None]-h[None])+np.sum(delta*delta,1)[:,None]
     err=ds[np.arange(len(rec)),tar];nn=ds.argmin(1);val=np.column_stack([err/den,err,np.all(CODES[nn][:,targets]==tc[:,targets],axis=1),np.all(CODES[nn][:,other]==tc[:,other],axis=1),np.all(CODES[nn]==tc,axis=1)]);results[method][i,si]=val.mean(0)
     if si<3:
      ex=~(np.isin(rec,[0,26])|np.isin(don,[0,26])|np.isin(tar,[0,26]));excluded[method][i,si]=val[ex].mean(0)
     # Saved float64 fixed examples; full ambient-space formula independent of low-space distance expansion.
     pick=np.linspace(0,len(rec)-1,3).astype(int);direct=np.sum((x[rec[pick],None]+delta[pick,None]@U.T-x[None])**2,-1)
     validation['full_distance_maxabs']=max(validation['full_distance_maxabs'],float(np.abs(direct-ds[pick]).max()))
     key=f'{name}_{i}_{si}_{method}';traces[key]=np.column_stack([rec[pick],don[pick],tar[pick],delta[pick],ds[pick,np.argmin(ds[pick],axis=1)],val[pick,0]]);trace_index.append([key,name,i,j,si,method])
     if method=='noedit':np.testing.assert_allclose(val[:,0],1,rtol=0,atol=1e-10)
    # Re-encode sequential edits without access to true coordinates; fixed one of the cases.
    if len(targets)==2:
     for method in Bs:
      b,inv=Bs[method][i],As[method][i];start=h[rec[0]];donoru=us[method][j][don[0]]
      def edit(v,k,goal):
       u=inv@(v-aa[i]);return v+b[:,2*k:2*k+2]@(goal[2*k:2*k+2]-u[2*k:2*k+2])
      f1,f2=targets;e12=edit(edit(start,f1,donoru),f2,donoru);e21=edit(edit(start,f2,donoru),f1,donoru)
      back=edit(edit(e12,f2,inv@(start-aa[i])),f1,inv@(start-aa[i]))
      validation['commutator_maxabs']=max(validation['commutator_maxabs'],float(np.abs(e12-e21).max()));validation['cycle_maxabs']=max(validation['cycle_maxabs'],float(np.abs(start-back).max()))
   if i%8==0:print(name,'scene',i,'done',flush=True)
  np.savez_compressed(o/(name+'_scene_metrics.npz'),**results);np.savez_compressed(o/(name+'_anchor_excluded.npz'),**excluded)
  rng=np.random.default_rng(20261013);draw=rng.integers(n//2,size=(5000,n//2));contrasts={}
  for ref in ['global6','wrong6','tworef']:
   diff=results['local6']-results[ref];pair=(diff[:n//2]+diff[n//2:])/2
   contrasts[ref]={'local_minus_reference':pair.mean(0).tolist(),'ci95_unadjusted':np.quantile(pair[draw].mean(1),[.025,.975],axis=0).tolist()}
  allresults[name]={'identity':'old tune/exploration, no independent confirmation','scene_count':n,'independent_pairs':n//2,'methods':{m:v.mean(0).tolist() for m,v in results.items()},'anchor_excluded_single':{m:v.mean(0).tolist() for m,v in excluded.items()},'contrasts':contrasts,'chart_stats':stats,'kernel_effective_neighbors':(1/np.sum(w*w,1)).tolist(),'nearest_descriptor_distance':np.sqrt(dist2.min(1)).tolist(),'chart_reconstruction_residual':chart_prediction_residual,'heldout_anova_diagnostic_only':anova(H)}
  sourcefiles.extend([r/dat/'dataset.npz',r/feat/'features.npz']);print(name,[(m,round(float(v[:,1,0].mean()),5)) for m,v in results.items()],flush=True)
 np.savez_compressed(o/'sample_edit_predictions.npz',**traces);dump(o/'sample_edit_index.json',trace_index)
 assert validation['full_distance_maxabs']<1e-8 and validation['inverse_alternate_maxabs']<1e-8
 dump(o/'results.json',{'metric_order':['relative_fullspace_error','absolute_fullspace_error','targets_nearest_correct','nontargets_preserved','all_correct'],'selection_order':[[0],[1],[2],[0,1],[0,2],[1,2]],'datasets':allresults})
 validation.update(status='PASS_NUMERICS',seconds=time.time()-t,sampled_edits=len(traces)*3,scope='fixed sample direct full5120 distances and alternate least-squares coordinate inverse; cycle/commute are algebraic invariants')
 dump(o/'validation.json',validation);dump(o/'source_hashes.json',{str(p):sha(p) for p in dict.fromkeys(sourcefiles)})
 shutil.copy2(__file__,o/pathlib.Path(__file__).name);dump(o/'manifest.json',{p.name:sha(p) for p in o.iterdir() if p.is_file()});print(validation,flush=True)
if __name__=='__main__':main()
