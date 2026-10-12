from pathlib import Path
import json,hashlib,numpy as np
r=Path(r'F:\2610\CVPR27');o=r/'runs/round2_crossblock'
rows=json.loads((o/'results.json').read_text());cfg=json.loads((o/'config.json').read_text())
assert hashlib.sha256((r/'scripts/round2_crossblock_probe.py').read_bytes()).hexdigest()==cfg['code_sha256']
assert hashlib.sha256((r/'scripts/round2_b_shared_private.py').read_bytes()).hexdigest()==cfg['ridge_code_sha256']
err=0
for p in o.glob('*.npz'):
 z=np.load(p); ds=p.name.split('_')[0]; cache=np.load(r/f'runs/e04_inputs/cache/{ds}_ijepa_r0.npz')
 assert np.array_equal(z['source_dev'],cache['source_dev'])
 assert np.isfinite(z['prediction']).all()
 names=z['target_names'].tolist(); ys=z['truth']; pred=z['prediction']; scores=1-((ys-pred)**2).sum(0)/((ys-ys.mean(0))**2).sum(0)
 filename=p.stem[len(ds)+1:]
 method=next(m for m in ['full_block4','white','sfa'] if filename.startswith(m+'_'))
 diagnostic=filename[len(method)+1:]
 for j,name in enumerate(names):
  row=next(v for v in rows if v['dataset']==ds and v['method']==method and v['diagnostic']==diagnostic and v['target']==name)
  err=max(err,abs(row['dev_r2']-scores[j]))
  assert row['coordinates']==z['indices'][j].tolist()
assert err<1e-10
(o/'validation.json').write_text(json.dumps({'all_180_metrics_recomputed':True,'max_r2_error':err,'indices_match_cache':True,'source_hashes_match':True,'scope':'supervised diagnostic; no new representation training'},indent=2))
report=r/'docs/round2_first_batch_20261011.txt'
text=report.read_text(encoding='utf-8-sig').replace('跨块紧凑性补充诊断待汇总','跨块紧凑性补充诊断已完成')
text+='''
补充：同维数跨块诊断（已完成，同一张H100）
在冻结white/SFA/full_block4上，仅用训练标签选择每目标top1/2/4/8坐标及最佳完整单4维块；tune独立选择ridge。此实验是监督读出定位，未训练新的无监督方法。同为4个输入坐标也不代表坐标搜索空间相等：自由top4比32个完整块有更多候选。
Pong差异在已有full_block4的另一个块（块3，0-based）可读到R²=.7932，单坐标=.7559，跨块top4=.8019；原公共父块是块4。因此Pong的首要问题是选错解释范围，而非必须跨多个块才能表达差异。
Pinball三个差异在各自最佳完整单4块的R²=.3986/.3062/.1825；同4坐标的全局选择=.4760/.5813/.3691，top8=.5732/.6923/.4364。它更支持差异分散在多个现有块的诊断，但尚未证明新的无监督重组能够自动找到这些方向。
共180行指标、30份预测，原始预测重算与源索引核对通过。保留所有三种变换的结果，不仅展示full_block4。
阶段下一步：先在完整表示中定义无标签候选分组与跨块搜索，再做局部/非正交四格；Pong需检验已有公共块与差异块能否稳定配对，Pinball需检查分散差异能否被紧凑重组。仍需独立片段确认，暂不宣称因素或因果机制已被无监督识别。
'''
report.write_text(text,encoding='utf-8')
index={'date':'2026-10-11','phase':'first_batch_completed','resource':{'host':'H100','gpu_count':1,'physical_gpu_index':0},'scope':'exploration_only','report':'docs/round2_first_batch_20261011.txt','completed':['data_and_protocol_audit','shared_baselines_and_pullback_ridge','A0_synthetic_controls','A1_global_parent_block_nonorthogonal','B0_common_contrast_information_audit','B1_minimal_hierarchy','supervised_budget_matched_crossblock_diagnostic'],'pending':['local_orthogonal_nonorthogonal_four_way','label_free_crossblock_regrouping','verified_unseen_confirmation','history_extension','matched_simulation_intervention'],'artifacts':[str(p.relative_to(r)) for p in [r/'runs/round2_a',r/'runs/round2_b',r/'runs/round2_audit',r/'runs/round2_evaluation',r/'runs/round2_crossblock',r/'runs/round2_execution']],'git_push':False}
(r/'docs/round2_experiment_index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
print('Crossblock verification and first-batch report complete',err)
