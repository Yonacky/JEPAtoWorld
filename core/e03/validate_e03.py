import pathlib,json,time,hashlib,tarfile
import numpy as np
R=pathlib.Path(__file__).parent;start=time.time()
while not (R/'COMPLETE.json').exists():
 if time.time()-start>10800:raise RuntimeError('run completion timeout; inspect run.log')
 time.sleep(10)
rows=json.loads((R/'forecasts.json').read_text());pr=json.loads((R/'probes.json').read_text());st=json.loads((R/'structures.json').read_text());checks=json.loads((R/'invariance_checks.json').read_text());count=0;worst=0.;cache={};basis={}
for ds in ['pong','pinball']:
 for enc in ['ijepa','random','dino','pixels']:
  for rep in range(3):
   tag=f'{ds}_{enc}_r{rep}';z=np.load(R/'cache'/f'{tag}.npz');cache[tag]={k:z[k] for k in ['dev','target_dev','source_dev']}
   assert np.all(np.diff(z['source_train'])==1) and np.all(np.diff(z['source_tune'])==1) and np.all(np.diff(z['source_dev'])==1)
   assert set(z['source_train']).isdisjoint(z['source_dev'])
for row in rows:
 tag=f"{row['dataset']}_{row['encoder']}_r{row['replica']}";key=tag+'_'+row['method'];z=cache[tag];h=row['horizon'];fit=np.load(R/row['model_file'])
 if key not in basis:
  b=np.load(R/'transforms'/(key+'.npz'));basis[key]=(b['past_basis'],b['future_basis'])
 ql,qr=basis[key];pred=(((z['dev'][:-h]@ql)-fit['input_mean'])@fit['weight']+fit['output_mean'])@qr.T;target=z['dev'][h:];assert np.isfinite(pred).all();perframe=np.mean((pred-target)**2,axis=1)
 assert np.array_equal(fit['source_indices'],z['source_dev'][h:]);assert np.all(z['source_dev'][h:]-z['source_dev'][:-h]==h)
 delta=max(float(np.max(np.abs(perframe-fit['per_frame_mse']))),abs(float(perframe.mean())-row['dev_mse']));worst=max(worst,delta);assert delta<1e-10
 if row['grouping']!='natural_schur':
  size=int(row['grouping']);mask=np.arange(128)[:,None]//size!=np.arange(128)[None,:]//size;assert np.max(np.abs(fit['weight'][mask]),initial=0)<1e-14
for row in pr:
 tag=f"{row['dataset']}_{row['encoder']}_r{row['replica']}";z=cache[tag];pp=np.load(R/row['predictions']);p=pp['prediction'];y=z['target_dev'];assert np.isfinite(p).all();assert np.array_equal(pp['source_indices'],z['source_dev']);gs=json.loads((R/(row['dataset']+'_factor_schema.json')).read_text())
 for g in gs:
  a,b=g['slice'];t=y[:,a:b];q=p[:,a:b];m=row['metrics'][g['name']]
  if g['type']=='categorical':
   tr=t.argmax(1);guess=q.argmax(1);v=float(np.mean([np.mean(guess[tr==k]==k) for k in np.unique(tr)]));assert abs(v-m['balanced_accuracy'])<1e-10
  else:
   den=np.sum((t-t.mean(0))**2)
   if den>1e-12:v=1-np.sum((q-t)**2)/den;assert abs(v-m['r2'])<1e-10
   else:assert m['r2'] is None
  count+=1
validation={'forecast_rows_reconstructed':len(rows),'probe_rows_recomputed':len(pr),'factor_metrics_recomputed':count,'forecast_reconstruction_max_error':worst,'all_index_and_block_mask_checks_passed':True,'dense_forecast_rotation_max_error':max(x['maxabs'] for x in checks if x['kind']=='dense_forecast_invariance'),'dense_probe_rotation_max_error':max(x['maxabs'] for x in checks if x['kind']=='dense_probe_invariance')};(R/'validation.json').write_text(json.dumps(validation,indent=2))
# Summaries preserve each segment and min/max, never call them independent-environment CIs.
summary={'probe':[],'forecast':[],'operator':[]}
for ds in ['pong','pinball']:
 keys=['ball_x','ball_y','paddle_left_y','paddle_right_y'] if ds=='pong' else ['ball_x','ball_y','paddle_left_y_pos','paddle_right_y_pos']
 for enc in ['ijepa','random','dino','pixels']:
  for method in json.loads((R/'protocol.json').read_text())['methods']:
   for budget in [1,4,8,'dense']:
    rr=sorted([x for x in pr if (x['dataset'],x['encoder'],x['method'],x['budget'])==(ds,enc,method,budget)],key=lambda x:x['replica']);values=[float(np.mean([x['metrics'][k]['r2'] for k in keys])) for x in rr];summary['probe'].append(dict(dataset=ds,encoder=enc,method=method,budget=budget,mean=float(np.mean(values)),minimum=min(values),maximum=max(values),segments=values))
   for h in [1,4,8]:
    for grouping in ['1','2','4','8','128']+(['natural_schur'] if method=='real_schur' else []):
     rr=sorted([x for x in rows if (x['dataset'],x['encoder'],x['method'],x['horizon'],x['grouping'])==(ds,enc,method,h,grouping)],key=lambda x:x['replica']);values=[x['dev_mse'] for x in rr];skills=[x['skill_vs_persistence'] for x in rr];summary['forecast'].append(dict(dataset=ds,encoder=enc,method=method,horizon=h,grouping=grouping,mse_mean=float(np.mean(values)),segments_mse=values,skill_mean=float(np.mean(skills)),segments_skill=skills,parameters=[x['parameters'] for x in rr]))
  rr=[x for x in st if x['dataset']==ds and x['encoder']==enc and x['method']=='real_schur'];summary['operator'].append(dict(dataset=ds,encoder=enc,antisymmetric_ratio=[x['antisymmetric_train_ratio'] for x in rr],schur_off_natural_block_energy=[x['off_natural_block_energy'] for x in rr],schur_natural_sizes=[x['natural_group_sizes'] for x in rr]))
(R/'summary.json').write_text(json.dumps(summary,indent=2))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
methods=['white','sfa','sfa_time_shuffle','sym_block4','full_block2','full_block4','full_block8','real_schur','lag_svd_dual'];labels=['White','SFA','Shuffle SFA','Sym block4','Full block2','Full block4','Full block8','Real Schur','Dual SVD']
fig,axs=plt.subplots(1,2,figsize=(15,5))
for ds,ax in zip(['pong','pinball'],axs):
 for enc,color in zip(['ijepa','random','dino','pixels'],['#145DA0','#DA7756','#40916C','#7C5CBF']):
  ss=[next(x for x in summary['probe'] if (x['dataset'],x['encoder'],x['method'],x['budget'])==(ds,enc,m,4)) for m in methods];val=np.array([x['mean'] for x in ss]);low=val-np.array([x['minimum'] for x in ss]);hi=np.array([x['maximum'] for x in ss])-val;ax.errorbar(np.arange(len(ss)),val,yerr=np.stack([low,hi]),fmt='o-',label=enc,color=color,capsize=2)
 ax.set_title(ds);ax.set_xticks(np.arange(len(methods)),labels,rotation=40,ha='right');ax.set_ylabel('Mean position R2, four selected coordinates');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Three new disjoint time segments; error bars are min-max, not confidence intervals');fig.tight_layout();fig.savefig(R/'new_segments_position.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(14,5))
for ds,ax in zip(['pong','pinball'],axs):
 for grouping,color in zip(['1','4','128'],['#DA7756','#145DA0','#40916C']):
  vals=[next(x['mse_mean'] for x in summary['forecast'] if (x['dataset'],x['encoder'],x['method'],x['horizon'],x['grouping'])==(ds,'ijepa',m,1,grouping)) for m in methods];ax.plot(vals,'o-',label='block '+grouping,color=color)
 ax.set_title(ds+' I-JEPA, one-step');ax.set_xticks(range(len(methods)),labels,rotation=40,ha='right');ax.set_ylabel('MSE in common whitened coordinates');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(R/'directional_forecast.png',dpi=150);plt.close(fig)
manifest={}
for folder in ['cache','transforms','probes','predictor_fits','features']:
 for p in (R/folder).glob('*'):
  if p.is_file():manifest[str(p.relative_to(R))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size}
(R/'remote_evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(R/'delivery.tar.gz','w:gz') as t:
 for p in R.iterdir():
  if p.is_file() and p.suffix in ['.py','.json','.log','.sh','.png']:t.add(p,arcname=str(p.relative_to(R.parent)),recursive=False)
 for p in (R/'features').glob('*.json'):t.add(p,arcname=str(p.relative_to(R.parent)),recursive=False)
sha=hashlib.sha256((R/'delivery.tar.gz').read_bytes()).hexdigest();(R/'delivery.sha256').write_text(sha);print('VALIDATED_AND_PACKAGED',validation,sha,flush=True)
