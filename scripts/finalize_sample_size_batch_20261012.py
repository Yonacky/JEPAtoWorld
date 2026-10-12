from pathlib import Path
import json,hashlib,re,zipfile
r=Path('F:/2610/CVPR27');names=['history_learning_curve_stove_20261012','history_learning_curve_drawer_20261012','history_learning_curve_summary_20261012','static_sample_scaling_20261012','libero_learning_curve_source_audit_20261012'];checked=[]
for n in names:
 d=r/'runs'/n;mf=d/('artifact_manifest.json' if (d/'artifact_manifest.json').exists() else 'manifest.json')
 for k,v in json.loads(mf.read_text(encoding='utf-8-sig')).items():
  rel=k.replace(chr(92),'/');p=r/rel if rel.split('/')[0] in ['runs','scripts','docs'] else d/rel;h=v['sha256'] if isinstance(v,dict) else v;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,str(p);checked.append({'path':str(p.relative_to(r)),'sha256':h})
report='''数据量假设验证汇总，2026-10-12
结论：原12条训练示范确实限制了部分历史和分块方法的表现。此前直接建议暂停部分路线偏早，应改成数据量条件下的判断。但扩到42条也没有让所有方法稳定超过简单基线，不能把全部问题归结为样本量。

实验：炉台取碗与抽屉取碗各50条示范；原4条tune、4条dev保持不动，train12/24/42。24为三个预先固定的新增子集，全部包含原12，不选择最佳。仅新增30条/任务编码，共新增2224帧，总3684帧，单H100 FP32；旧20条的特征及全部样本信息逐值不变。固定方法、维数和正则网格；所有变换只用对应train重拟合。未来目标坐标及状态误差尺度用原12train冻结，避免规模变化偷偷改变预测任务。两任务分别训练、旧dev反复使用，属于探索。

历史路线：完整历史的后向位移固定尺度MSE，炉台 .258139→.117819，抽屉 .198107→.093577，下降约54%/53%。未来4帧表示预测分别 .924196→.704936、.941856→.663228。RRR64同样受益，未来4帧1.003478→.718642、1.014649→.668876；42条时与当前64的.687256/.662797接近，部分8帧指标小幅互有胜负且区间跨零，未建立稳定未来预测优势。完整历史42条的位移优于PCA/RRR压缩，提示样本量与压缩损失都存在。完整历史256维与当前64维容量不同，结果不是同维胜出；位移是观测后向差，不是受控物理动力学。

静态路线：分块top8位置MSE在两任务.30390→.14920、.29892→.15908，改善实质。但42条white/SFA分别为炉台.05987/.04772、抽屉.09111/.07293，分块仍落后。夹爪并无一致增益；炉台42分块还比三个24子集差，不能称曲线单调或已经饱和。固定尺度MSE和原字段MSE均保存，避免轴加权掩盖退化。

核验：两任务旧数据、固定split、嵌套子集与历史/静态子集一致；180静态预测独立误差≤2.39e-15，历史独立求解≤4.85e-13。图中24误差条只是三个子集范围，不是置信区间。只有每任务4个旧dev，配对demo区间也仅条件于它们，不能作充分泛化证明。代码、所有子集配置、预测、图和清单均保存F盘，未推送GitHub。

计划修订：保留原G批次负结果但收回基于12demo直接否定历史/分块的解释。将足量训练与学习曲线设为今后路线筛选的前置条件。当前两HDF5已用完除固定tune/dev之外的42demo，不再通过提高抽帧密度冒充更多独立轨迹。下一规模扩展须增加独立仿真rollout或额外任务，并留出更大、未用于选择的评价集合；与原G-4输入保留、G-5动作物理干预和G-6解释功能包兼容，暂不声称更多数据必然取得方法优势。
'''
(r/'docs/sample_size_batch_report_20261012.txt').write_text(report,encoding='utf-8')
p=r/'docs/MASTER_PLAN.md';s=p.read_text(encoding='utf-8-sig');a=s.index('当前定位：');b=s.index('\n\n',a);s=s[:a]+'当前定位：S-0至S-3数据量验证已完成。固定旧评估demo后，训练12→42显著改善完整历史位移和部分分块读出，先前基于12demo暂停路线的判断需附加样本量条件；42demo尚未建立普遍方法优势。未来路线筛选先检查学习曲线，再推进G-4/5/6信息保留、真实动力学及解释功能包。'+s[b:];lines=s.splitlines()
for i,l in enumerate(lines):
 if l.startswith('| S-0 |'):lines[i]='| S-0 | G-0 / R2-4 | 数据量假设统一核验 | 已完成并核验备份；2026-10-12 | 主智能体 | 固定4tune/4dev，12/24/42两任务；数据不足解释部分失败但非全部；[汇总](sample_size_batch_report_20261012.txt) |'
 if l.startswith('| R2-4 |'):lines[i]='| R2-4 | 合并检查并冻结候选 | 路线及数据量条件首批完成，整体部分完成 | [数据量验证](sample_size_batch_report_20261012.txt)；不再仅凭12demo否定路线 | 独立rollout扩大训练/评估；信息保留及受控干预后冻结候选 |'
 if l.startswith('| R2-5 |'):lines[i]='| R2-5 | 历史与动力学解释 | G-1与S-2学习曲线完成，物理验证待做 | [历史曲线](history_learning_curve_report_20261012.txt)；42demo完整历史位移明显改善、未来追近当前帧 | 更大独立评估及G-5动作/物理分支，禁止以位移读出替代物理因果结论 |'
s='\n'.join(lines)+'\n';s=s.replace('## 维护与协作规则','| 2026-10-12 | S-0至S-3数据量验证 | 已完成；12→42历史位移误差下降约54%/53%，分块部分改善 | 修订小样本路线判断，增加独立数据后再筛选 | [汇总](sample_size_batch_report_20261012.txt) |\n\n## 维护与协作规则');s+='\n| 2026-10-12 | S系列完成并修订路线筛选条件 | 用户数据量假设获部分支持，原方法失败不能全归方法 | 保留原负结果，增加独立示范而非帧密度；旧dev不称确认 |\n';s=re.sub(r'(?m)(^\|[^\n]*\n)(?:[ \t]*\n)+(?=\|)',r'\1',s);p.write_text(s,encoding='utf-8')
for t in re.findall(r'\]\(([^)]+)\)',s):
 if '://' not in t and not t.startswith('#'):assert (p.parent/t).resolve().exists(),t
(r/'runs/sample_size_batch_validation_20261012.json').write_text(json.dumps({'status':'PASS','count':len(checked),'files':checked,'plan_links':'PASS'},indent=2))
files=set()
for n in names+['libero_learning_curve_20261012','libero_learning_curve_preparation_20261012']:files.update(f for f in (r/'runs'/n).rglob('*') if f.is_file() and f.suffix!='.zip')
for folder,patterns in [('scripts',['*learning*20261012*','*scaling*20261012*']),('docs',['*learning*20261012*','*scaling*20261012*','sample_size_batch_report_20261012.txt','MASTER_PLAN.md']),('runs',['learning_curve*validation*','sample_size_batch_validation_20261012.json'])]:
 for pat in patterns:files.update(f for f in (r/folder).glob(pat) if f.is_file())
a=r/'experiments/sample_size_batch_20261012.zip';assert not a.exists()
with zipfile.ZipFile(a,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(files):z.write(f,f.relative_to(r).as_posix())
with zipfile.ZipFile(a) as z:
 assert z.testzip() is None
 for f in files:assert hashlib.sha256(z.read(f.relative_to(r).as_posix())).digest()==hashlib.sha256(f.read_bytes()).digest()
meta={'status':'PASS','files':len(files),'bytes':a.stat().st_size,'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'verified_manifest_entries':len(checked)};(r/'runs/sample_size_batch_backup_20261012.json').write_text(json.dumps(meta,indent=2));print(meta)
