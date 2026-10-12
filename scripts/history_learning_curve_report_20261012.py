from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('F:/2610/CVPR27');OUT=R/'runs/history_learning_curve_summary_20261012';OUT.mkdir(exist_ok=True);methods=['current64','history256','history_pca64','history_rrr64'];colors=['#244f8f','#dc7b23','#269578','#aa4b82'];tasks=['stove','drawer'];targets=['future4frames','backward_displacement'];fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True);summary=[];lines=['S-2 固定目标空间历史路线学习曲线；2026-10-12。','结论：数据量不足是此前部分负结果的重要因素。扩到42条训练demo后，完整历史的位移读出和未来表示预测明显改善，预测导向压缩也改善；未来预测在42条时追平或小幅互有胜负，没有建立稳定超过当前帧的优势。不能把此前12条数据上的结果概括成历史路线无效，也不能把42条结果当作完整动力学证明。','每任务保留原tune4/dev4，训练12/24/42。12及42各算一次，24按seed20261012/13/14三个新增子集，不选最佳子集。三个24范围仅表示子集选择范围，不是置信区间。仍是反复使用的4条dev，不是新确认；各任务分别拟合，非零样本跨任务泛化。','共同目标空间是旧12train缓存全部帧的PCA64白化映射，预测未来4/8原始帧；该映射和各目标原12共同窗口尺度在所有规模冻结。每规模输入PCA64重新仅train拟合；历史256、PCA历史64、RRR64，lambda=.01固定。RRR也使用该冻结未来目标，是任务对齐学习与可比性控制。','本轮tune改用demo等权选择正则（G-1为帧加权tune），评价同样demo等权。因此12条基线已重新计算，只比较S-2内部曲线，不与旧G-1数值直接拼接。拟合仍帧加权，长demo训练贡献较多。','状态/位移同时保存原单位平方误差与固定旧12train尺度MSE；位移是后向4原始帧位置差，不是速度。JSON字段macro_physical_mse对未来组实为固定latent坐标原尺度误差，并非物理单位；未来图采用冻结尺度归一MSE。']
for col,task in enumerate(tasks):
 o=R/f'runs/history_learning_curve_{task}_20261012';res=json.loads((o/'results.json').read_text());rows=res['rows'];cfg=json.loads((o/'config.json').read_text());val=json.loads((o/'validation.json').read_text());lines.append('\n'+task+'：method / 位移12→42 / future4原帧12→42 / future8原帧12→42')
 for method in methods:
  vals=[]
  for target in ['backward_displacement','future4frames','future8frames']:
   vals.append([next(r['macro_std_mse'] for r in rows if r['method']==method and r['target']==target and r['subset']==s) for s in ['n12','n42']])
  lines.append(method+' / '+' / '.join(f'{a:.6f}→{b:.6f}' for a,b in vals))
 for row,target in enumerate(targets):
  ax=axes[row,col]
  for method,color in zip(methods,colors):
   p12=next(r['macro_std_mse'] for r in rows if r['method']==method and r['target']==target and r['subset']=='n12');p42=next(r['macro_std_mse'] for r in rows if r['method']==method and r['target']==target and r['subset']=='n42');p24=np.array([r['macro_std_mse'] for r in rows if r['method']==method and r['target']==target and r['ntrain']==24]);ax.plot([12,24,42],[p12,p24.mean(),p42],marker='o',color=color,label=method);ax.errorbar([24],[p24.mean()],yerr=[[p24.mean()-p24.min()],[p24.max()-p24.mean()]],capsize=5,color=color);summary.append({'task':task,'method':method,'target':target,'n12':p12,'n24_mean':float(p24.mean()),'n24_min':float(p24.min()),'n24_max':float(p24.max()),'n42':p42})
  ax.grid(alpha=.2);ax.set_title(task+' / '+('future +4 frames' if row==0 else 'backward displacement'));ax.set_xticks([12,24,42]);ax.set_ylabel('Fixed-scale demo-macro MSE');ax.set_xlabel('Training demonstrations')
 for c in res['n12_vs_n42']:
  if c['target'] in targets:lines.append(f"{c['method']} {c['target']} 12减42改善={c['n12_minus_n42_positive_improves']:.6f}，配对demo条件CI={c['paired_demo_ci95']}")
 lines.append('独立直接解最大差='+str(max(c['direct_error'] for c in val['checks']))+'；落盘评分复算差='+str(val['saved_metric_max_error'])+'；旧特征差='+str(cfg['original_old_feature_max_error']))
 files=list(o.iterdir());(o/'artifact_manifest.json').write_text(json.dumps({str(p.relative_to(R)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files},indent=2))
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False);fig.text(.5,-.005,'Bars at 24: range across 3 nested subsets, NOT confidence intervals. Fixed 4 dev demos per task.',ha='center',fontsize=10);fig.tight_layout(rect=[0,.02,1,.96]);fig.savefig(OUT/'history_learning_curves.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'history_learning_curves.pdf',bbox_inches='tight');plt.close(fig)
lines+=['\n解释：完整历史需要更多训练数据才能估计较多方向；42条时位移改善较明确，但未来表示误差尚未明显优于当前帧。PCA64的stove位移反而随数据增加变差，提示重构方差目标不保证保留运动信息。RRR比PCA未来预测更好，与强单帧追平或小幅互有胜负，因此样本量与压缩目标都影响结论，不能只归因其中之一。','未追加模型/超参搜索，也未用dev选择训练子集。训练规模只覆盖12至42，不能外推成已收敛或更多数据必然胜出。状态和表示预测仍不能替代动作/质量/摩擦干预。','图：runs/history_learning_curve_summary_20261012/history_learning_curves.png及.pdf；两任务完整预测、配置、实际demo名单、固定目标映射及数值核验均保留。']
(R/'docs/history_learning_curve_report_20261012.txt').write_text('\n'.join(lines),encoding='utf8');(OUT/'plot_values.json').write_text(json.dumps(summary,indent=2))
