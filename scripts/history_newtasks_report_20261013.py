from pathlib import Path
import json,hashlib,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('F:/2610/CVPR27');O=R/'runs/history_newtasks_summary_20261013';O.mkdir(exist_ok=True);tasks=['cabinet','cookie','ramekin'];methods=['current64','history256','history_pca64','history_rrr64'];targets=['backward_displacement','future4frames','future8frames'];allrows=[];comparisons=[];validations=[];rng=np.random.default_rng(20261013);boot=rng.integers(10,size=(2000,10));fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
for j,task in enumerate(tasks):
 d=R/f'runs/history_newtask_{task}_20261013';res=json.loads((d/'results.json').read_text());v=json.loads((d/'validation.json').read_text());cfg=json.loads((d/'config.json').read_text());rows=res['rows'];allrows.extend([{'task':task,**r} for r in rows]);validations.append({'task':task,'max_direct':max(c['direct_error'] for c in v['checks']),'saved_metric_error':v['saved_metric_max_error'],'tune_dev_windows':cfg['counts']})
 for method,ref in [('history256','current64'),('history_pca64','current64'),('history_rrr64','current64'),('history256','history_pca64'),('history256','history_rrr64')]:
  for target in targets:
   a=next(r for r in rows if r['subset']=='n30' and r['method']==method and r['target']==target);b=next(r for r in rows if r['subset']=='n30' and r['method']==ref and r['target']==target);dd=np.array([b['demo_std_mse'][k]-a['demo_std_mse'][k] for k in sorted(a['demo_std_mse'])]);comparisons.append({'task':task,'method':method,'reference':ref,'target':target,'positive_improves':float(dd.mean()),'demo_ci95':np.percentile(dd[boot].mean(1),[2.5,97.5]).tolist(),'paired_deltas':dd.tolist()})
 for i,target in enumerate(['future4frames','backward_displacement']):
  ax=axes[i,j]
  for method,color in zip(methods,['#244f8f','#dc7b23','#269578','#aa4b82']):
   yy=[next(r['macro_std_mse'] for r in rows if r['subset']==s and r['method']==method and r['target']==target) for s in ['n12','n30']];ax.plot([12,30],yy,marker='o',label=method,color=color)
  ax.set_title(task+' / '+('future +4 frames' if i==0 else 'backward displacement'));ax.set_xticks([12,30]);ax.set_xlabel('Training demos');ax.set_ylabel('Fixed-scale demo-macro MSE');ax.grid(alpha=.2)
 lines=[f'T-2 {task}新任务历史路线复现；各30/10/10完整demo；12/30嵌套，首12固定目标与尺度。','同demo stride4四帧历史，未来4/8原始帧；4路线原预算，不新增调参。每任务重新拟合，非零样本迁移。下列均为10demo宏平均固定尺度MSE，越低越好。']
 for size in ['n12','n30']:
  lines.append(size+'：method / 位置 / 夹爪 / 位移 / 夹爪变化 / future4 / future8')
  for method in methods:
   vals=[next(r['macro_std_mse'] for r in rows if r['subset']==size and r['method']==method and r['target']==t) for t in ['position','gripper','backward_displacement','gripper_change','future4frames','future8frames']];lines.append(method+' / '+' / '.join(f'{v:.6f}' for v in vals))
 lines+=['30train路线配对差（参照减方法，正值改善）；2000次demo bootstrap，条件于本任务10dev，不作多重比较或任务总体推断。']+[str(c) for c in comparisons if c['task']==task];lines.append(str(validations[-1]));(R/f'docs/history_newtask_{task}_20261013.txt').write_text('\n'.join(lines),encoding='utf8')
# Task-stratified equal-task bootstrap: resample demos independently inside each of the fixed3 tasks.
aggregate=[]
for method,ref in [('history256','current64'),('history_pca64','current64'),('history_rrr64','current64'),('history256','history_pca64'),('history256','history_rrr64')]:
 for target in targets:
  group=[c for c in comparisons if c['method']==method and c['reference']==ref and c['target']==target];samples=[]
  for c in group:
   ix=rng.integers(10,size=(2000,10));samples.append(np.array(c['paired_deltas'])[ix].mean(1))
  aggregate.append({'method':method,'reference':ref,'target':target,'task_equal_delta':float(np.mean([c['positive_improves'] for c in group])),'within_task_bootstrap_ci95':np.percentile(np.mean(samples,axis=0),[2.5,97.5]).tolist()})
(O/'comparisons.json').write_text(json.dumps({'per_task':comparisons,'task_equal':aggregate,'validation':validations},indent=2));h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=4,frameon=False);fig.text(.5,.012,'Frozen methods; 10 held-out demos per task. Separate training per task; not zero-shot transfer.',ha='center');fig.tight_layout(rect=[0,.04,1,.93]);fig.savefig(O/'newtask_history_curves.png',dpi=160);fig.savefig(O/'newtask_history_curves.pdf');plt.close(fig)
lines=['T-2 三个新增LIBERO任务固定历史路线复现；2026-10-13。','结论：完整历史改善后向位移读出的趋势在三个新增任务复现；增加训练demo从12到30后完整历史继续改善，并优于两种64维压缩。未来表示预测也随训练量改善，但30条时仍未超过当前帧：不能把历史位移收益推广成未来世界模型优势。','cabinet/cookie/ramekin各50条完整demo，固定30train/10tune/10dev，三个任务合计30条dev。方法在看新任务结果前冻结，各task分别训练，非同一模型的跨task泛化或零样本迁移。所有窗口同demo/split，stride4；首12训练帧PCA64冻结未来坐标、首12共同窗口冻结状态尺度，12与30可比。输入PCA和历史压缩仅各train拟合，RRR lambda=.01，ridge网格不变，tune/dev demo等权。','位移目标两端都在历史输入窗口内，是回顾性读出而非未来状态预测；也不是真实速度或物理因果；未来是冻结JEPA64维坐标，不是未来状态。结果macro_physical_mse对状态为原单位MSE，对future是latent原尺度，不能统称物理量。','按任务分列30train MSE：current / history256 / PCA64 / RRR64。']
for task in tasks:
 for target in targets:
  vals=[next(r['macro_std_mse'] for r in allrows if r['task']==task and r['subset']=='n30' and r['method']==m and r['target']==target) for m in methods];lines.append(task+' '+target+' : '+' / '.join(f'{v:.6f}' for v in vals))
lines+=['\n三任务等权、任务内demo重采样的路线差（参照减方法正值改善；条件于已选3任务，不能当广泛任务总体CI）：']+[str(a) for a in aggregate];lines+=['\n独立直接线性求解与保存预测评分复算：']+[str(v) for v in validations];lines+=['状态变化读出收益有新任务复现支持，但完整历史256与压缩64存在容量差，不证明新子空间方法优越。保留完整历史作为诊断参照；压缩仍丢失运动信息，未来预测未见历史的稳定增益。后续应转向动作/物理干预证据，不按这批dev继续调参。','各任务报告docs/history_newtask_{cabinet,cookie,ramekin}_20261013.txt；结果runs/history_newtask_*_20261013；图runs/history_newtasks_summary_20261013/newtask_history_curves.png及PDF。']
(R/'docs/history_newtasks_report_20261013.txt').write_text('\n'.join(lines),encoding='utf8')
for task in tasks:
 d=R/f'runs/history_newtask_{task}_20261013';files=[p for p in d.iterdir() if p.name!='artifact_manifest.json']+[R/'scripts/history_new_tasks_20261012.py',R/f'docs/history_newtask_{task}_20261013.txt'];(d/'artifact_manifest.json').write_text(json.dumps({str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files},indent=2))
print(json.dumps(aggregate,indent=2))
