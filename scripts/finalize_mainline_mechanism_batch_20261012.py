from pathlib import Path
import json,hashlib,re,zipfile
r=Path('F:/2610/CVPR27');checked=[]
folders=[('mainline_predictive_history_20261012','artifact_manifest.json'),('mainline_predictive_subspace_20261012','manifest.json'),('intervention_20261012/target_supervised_reference','artifact_manifest.json'),('intervention_20261012/target_supervised_reference_stable','artifact_manifest.json')]
for name,mf in folders:
 d=r/'runs'/name
 for key,val in json.loads((d/mf).read_text(encoding='utf-8-sig')).items():
  rel=key.replace(chr(92),'/');p=(r/rel) if rel.split('/')[0] in ['runs','docs','scripts'] else d/rel
  h=val['sha256'] if isinstance(val,dict) else val
  assert hashlib.sha256(p.read_bytes()).hexdigest()==h,str(p)
  checked.append({'path':str(p.relative_to(r)),'sha256':h})
report='''多主线机制探索第二批，2026-10-12
本轮完成H-3、H-4、B-4及N-4，均已保存F盘。三个Astra子智能体参与，全部新拟合在CPU完成，未占额外GPU，未推送GitHub。旧数据复用属于探索，不是新确认。

历史线：固定128维，用四帧预测未来1/2/4帧的降秩岭压缩，相比PCA改善两环境未来表示预测。Pinball动力读出R² .1038→.2066，Pong .0682→.0753且增益区间跨零。仍低于全512维历史的.3302/.1589，历史重构变差，未来预测未一致胜当前单帧128。只支持任务对齐压缩能部分修复PCA丢失，不能声称动力因素已经恢复。根另用Cholesky白化加SVD独立实现重算，权重最大误差5.73e-15。

差异线：无因素标签的多滞后SVD允许连续旋转，但top8覆盖的联合差异训练关联能量仅Pong1.57%、Pinball19.95%，而部分差异在更后谱方向仍可读出。该关联能量是固定训练规范下协方差平方份额，不是因果贡献。输入旋转及标签尺度/Helmert基变换核验通过。因此问题不只是坐标轴选错，而是此具体总体预测能量排序与差异保留不对齐。去掉首预测方向仍不稳定，不能泛化成所有时间目标失败。

编辑线：更强的匹配目标监督，真实目标相对误差训练，并要求train每因素保留率不少于joint0的95%。固定四惩罚权重均在第37步取得相同最佳可行点，是约束激活前相同路径，不是四次独立成功。tune颜色误差.7321→.6757，但最近邻目标正确率.7402→.7188，保留率.7168→.6798，且不满足训练阈值；非目标保留率仍为1。无约束颜色误差更低.6020而恢复率降至.6583。该结果再次说明表示距离下降不能单独证明因素编辑改善，均值能量约束也不能保证tune目标恢复。它是有监督参考，不是无监督发现或理论上界。首次Adam投影优化回升已完整保留；改Armijo流形梯度后目标单调，数值核验通过，但未证明全局最优或受约束问题已解到最优。

来源审计：服务器Pong250000/Pinball150000帧原始trainNPZ哈希匹配既有缓存来源，原始文件本身确实没有episode/reset/done。两个settings已备份，但没有生成seed/commit及自然重采样标记。只能保留连续索引窗口结论，不能自动等同物理连续轨迹。

下一步顺序：编辑线先检查逐场景/颜色等级保留与真实目标识别间隔，区分平均约束的掩盖效应和目标不匹配；该机制没解决前不增加新确认集。历史线下一独立批次需要可追溯生成器和显式reset原因，再验证压缩收益。差异线保留完整谱与原块参照，下一目标需避免只按总体预测能量截断。R2-2至R2-6的整体任务均未因本批运行完成而标为完成。
'''
(r/'docs/mainline_mechanism_batch_report_20261012.txt').write_text(report,encoding='utf-8')
p=r/'docs/MASTER_PLAN.md';s=p.read_text(encoding='utf-8-sig');start=s.index('当前定位：');end=s.index('\n\n',start);s=s[:start]+'当前定位：最新H-3/H-4、B-4、N-4机制批次完成并核验。预测导向压缩部分改善同维历史保留；最高预测能量排序遗漏因素差异；匹配目标监督编辑仍有距离与恢复率取舍。全部为旧数据探索，原两参照基线及负结果保留。整体仍沿六步总路线和四步干预增补推进。'+s[end:]
lines=s.splitlines()
for i,line in enumerate(lines):
 if line.startswith('| R2-3 |'):lines[i]='| R2-3 | 路线B：公共与差异成分 | 部分完成，B-3/B-4已核验 | [预测子空间报告](mainline_predictive_subspace_report_20261012.txt)；最高预测能量排序遗漏差异，后谱仍含信息 | 不只按总体预测性截断；新机制及独立片段验证仍待做 |'
 if line.startswith('| R2-5 |'):lines[i]='| R2-5 | 历史与动力学解释 | H-1至H-4完成探索与来源审计，系统验证待做 | [预测历史报告](mainline_predictive_history_report_20261012.txt)；[原始来源](history_original_source_report_20261012.txt) | 可追溯生成与显式重采样标记、新轨迹复现、动作与物理参数干预 |'
 if line.startswith('| I-4 |'):lines[i]='| I-4 | 迁移与可组合编辑 | 部分完成，N-4仍有目标取舍 | [两参照报告](intervention_tworef_report_20261012.txt)；[监督参考](target_supervised_reference_report_20261012.txt) | 逐场景目标保留和识别间隔、外推失败；未见视角/物体和动力学仍待做 |'
s='\n'.join(lines)+'\n';row='| 2026-10-12 | H-3/H-4、B-4、N-4机制批次 | 已完成并核验；保留压缩/预测排序/编辑目标取舍与首次优化失败 | 机制未闭合，不新增确认或宣称因素独立 | [汇总](mainline_mechanism_batch_report_20261012.txt) |\n\n';s=s.replace('## 维护与协作规则',row+'## 维护与协作规则');s+='\n| 2026-10-12 | 根合并机制批次与原始来源核验 | 延续多主线推进，区分程序核验与科学假设 | 各父计划仍部分完成；所有候选及失败保留，未推送 |\n';s=re.sub(r'(?m)(^\|[^\n]*\n)(?:[ \t]*\n)+(?=\|)',r'\1',s);p.write_text(s,encoding='utf-8')
for target in re.findall(r'\]\(([^)]+)\)',s):
 if '://' not in target and not target.startswith('#'):assert (p.parent/target).resolve().exists(),target
v=r/'runs/mainline_mechanism_validation_20261012.json';v.write_text(json.dumps({'status':'PASS','manifest_files_verified':len(checked),'files':checked,'plan_links':'PASS'},indent=2))
files=set()
for name,_ in folders:
 files.update(f for f in (r/'runs'/name).rglob('*') if f.is_file() and f.suffix!='.zip')
files.update(f for f in (r/'runs/history_source_audit_20261012').rglob('*') if f.is_file())
for rel in ['docs/MASTER_PLAN.md','docs/mainline_mechanism_batch_report_20261012.txt','docs/mainline_predictive_history_report_20261012.txt','docs/mainline_predictive_subspace_report_20261012.txt','docs/history_original_source_report_20261012.txt','docs/target_supervised_reference_report_20261012.txt','docs/n4_method_audit_20261012.txt','scripts/mainline_predictive_history_20261012.py','scripts/mainline_predictive_subspace_20261012.py','scripts/fit_target_supervised_reference_20261012.py','scripts/audit_predictive_history_root_20261012.py','runs/predictive_history_root_audit_20261012.json','runs/mainline_mechanism_validation_20261012.json']:
 files.add(r/rel)
a=r/'experiments/mainline_mechanism_batch_20261012.zip';assert not a.exists()
with zipfile.ZipFile(a,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(files):z.write(f,f.relative_to(r).as_posix())
with zipfile.ZipFile(a) as z:
 assert z.testzip() is None
 for f in files:assert hashlib.sha256(z.read(f.relative_to(r).as_posix())).digest()==hashlib.sha256(f.read_bytes()).digest()
meta={'status':'PASS','archive':str(a),'bytes':a.stat().st_size,'files':len(files),'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'verified_manifest_files':len(checked)};(r/'runs/mainline_mechanism_backup_20261012.json').write_text(json.dumps(meta,indent=2));print(meta)
