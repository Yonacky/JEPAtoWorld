from pathlib import Path
import json,hashlib,re,zipfile
r=Path('F:/2610/CVPR27');names=['history_newtask_cabinet_20261013','history_newtask_cookie_20261013','history_newtask_ramekin_20261013','history_newtasks_summary_20261013','static_newtask_replication_20261013','libero_validation_source_audit_20261013'];checked=[]
for n in names:
 d=r/'runs'/n;mf=d/('artifact_manifest.json' if (d/'artifact_manifest.json').exists() else 'manifest.json')
 for k,v in json.loads(mf.read_text(encoding='utf-8-sig')).items():
  rel=k.replace(chr(92),'/');p=r/rel if rel.split('/')[0] in ['runs','scripts','docs'] else d/rel;h=v['sha256'] if isinstance(v,dict) else v;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,str(p);checked.append({'path':str(p.relative_to(r)),'sha256':h})
report='''新增任务扩大评估汇总，T批次，执行日期2026-10-12，划分种子/文件标识20261013
本轮完成三个此前未用于本轮方法开发的任务：木柜上、饼干盒上、小碗上取碗。各50条demo，固定30train/10tune/10dev，共150条示范4489帧，新增30条评估示范。12/30嵌套训练，方法/维数/超参网格在查看新任务结果前冻结；单H100提取FP32表示，全部任务分别训练评价，不能称同模型零样本迁移。数据仍来自LIBERO spatial同一家族，源文件此前可能被审计过。

主要结论一：历史对回顾性运动读出的帮助跨三个场景复现。30条train时，后向位移固定尺度MSE从当前帧到完整历史：木柜.3054→.1882，饼干盒.2353→.0794，小碗.3622→.1470。完整历史自身从12到30条train分别.2617→.1882、.1577→.0794、.2463→.1470，样本量收益方向一致。位移两端均在历史输入窗口内，是回顾性读出，不是未来状态预测或物理因果证据；完整历史256维与当前64维容量不同。两种64维历史压缩也有部分收益，但未保住完整历史信息。

主要结论二：未来预测收益仍不能从位移读出推出。30train的未来4原始帧表示误差，当前/完整历史分别为木柜.6924/.7102、饼干盒.7124/.7543、小碗.7288/.7791。未来8帧也同向，所有三task的完整历史与两种压缩均未超过当前帧。训练量增加能降低误差，但未改变这三个新任务的相对格局。目标是首12train冻结的JEPA坐标，没有动作条件，不能将结果泛化为所有世界模型或所有动态解释方案无效。

主要结论三：静态分块随数据增加的改善并非普遍。top8位置MSE 12→30，木柜.2359→.2807退化，饼干盒.2550→.2091、小碗.3106→.2689改善；30条时均落后white/SFA。完整64维夹爪三task均改善，说明“更多数据改善读出”和“少数坐标能否集中语义”必须区分。保留简单基线与所有局部正/负结果，不再仅凭小训练集淘汰路线，也不把样本增加当作紧凑性问题的充分解法。

核验与限制：三task的历史/静态12/30训练顺序相同，训练、tune、dev互斥且时序窗不跨demo；源/特征hash与索引通过。历史独立正规方程重算误差≤1.53e-13；静态72组预测另有直接解核验。按完整demo评分，不把帧当独立样本。任务内bootstrap仅条件于固定三个任务，不能当所有任务总体区间。历史与静态的crop/尺度/标签选择规范不同，不直接跨分支比较绝对MSE。必要代码、全部预测、图表和失败记录均F盘保存，未推送GitHub。

下一步按原G计划推进：把历史保留作为基线与消融；G-4研究空间输入/压缩是否丢掉目标信息，G-5以同初态动作及物理参数分支检验真正未来状态差异，G-6联合目标恢复/非目标保持验证解释功能。暂不再追加相似任务来追逐单个均值。论文当前可支持的阶段结论是样本量影响估计、历史保留帮助回顾运动信息，而紧凑组织和未来预测仍需要不同证据。
'''
(r/'docs/newtask_batch_report_20261013.txt').write_text(report,encoding='utf-8')
p=r/'docs/MASTER_PLAN.md';s=p.read_text(encoding='utf-8-sig');a=s.index('当前定位：');b=s.index('\n\n',a);s=s[:a]+'当前定位：T-0至T-3三新增任务评估完成，新增30个dev demo。历史回顾性位移收益和样本量效应在三任务复现，但未来表示预测仍未胜当前帧；静态分块增量收益两任务改善、一任务退化。保留white/SFA与完整历史基线，转回G-4/5/6信息保留、动作物理分支和解释功能；不把更多数据或回顾读出当因素独立证据。'+s[b:];lines=s.splitlines()
for i,l in enumerate(lines):
 if l.startswith('| T-0 |'):lines[i]='| T-0 | R2-4 / R2-5 | 扩大独立评估协调和路线结论复核 | 已完成并核验备份 | 主智能体 | 三任务150demo/4489帧/30dev；冻结12/30、同子集与源核验；[汇总](newtask_batch_report_20261013.txt) |'
 if l.startswith('| R2-5 |'):lines[i]='| R2-5 | 历史与动力学解释 | S数据量与T新任务复现完成，物理验证待做 | [T批次](newtask_batch_report_20261013.txt)；历史回顾位移收益复现，未来表示预测仍未胜当前 | G-5动作时间对齐、同初态未来状态/物理分支，区分回顾与预测 |'
s='\n'.join(lines)+'\n';s=s.replace('## 维护与协作规则','| 2026-10-12 | T-0至T-3三新增任务验证 | 已完成；150demo、30dev、4489帧，历史位移收益复现、未来仍无优势、分块并非普遍改善 | 保留足量基线，回归输入/物理/功能三包 | [汇总](newtask_batch_report_20261013.txt) |\n\n## 维护与协作规则');s+='\n| 2026-10-12 | T系列完成 | 扩大评估后区分回顾读出、未来预测与紧凑组织 | 三任务分别fit，同spatial家族限制保留，原R2计划仍未全部完成 |\n';s=re.sub(r'(?m)(^\|[^\n]*\n)(?:[ \t]*\n)+(?=\|)',r'\1',s);p.write_text(s,encoding='utf-8')
for t in re.findall(r'\]\(([^)]+)\)',s):
 if '://' not in t and not t.startswith('#'):assert (p.parent/t).resolve().exists(),t
(r/'runs/newtask_batch_validation_20261013.json').write_text(json.dumps({'status':'PASS','count':len(checked),'files':checked,'plan_links':'PASS'},indent=2));files=set()
for n in names+['libero_validation_tasks_20261013','libero_validation_preparation_20261013']:files.update(f for f in (r/'runs'/n).rglob('*') if f.is_file() and f.suffix!='.zip')
for folder,patterns in [('scripts',['*newtask*','*new_tasks*','*libero_validation*']),('docs',['*newtask*','*libero_validation*','MASTER_PLAN.md']),('runs',['newtask*validation*.json'])]:
 for pat in patterns:files.update(f for f in (r/folder).glob(pat) if f.is_file())
a=r/'experiments/newtask_batch_20261013.zip';assert not a.exists()
with zipfile.ZipFile(a,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(files):z.write(f,f.relative_to(r).as_posix())
with zipfile.ZipFile(a) as z:
 assert z.testzip() is None
 for f in files:assert hashlib.sha256(z.read(f.relative_to(r).as_posix())).digest()==hashlib.sha256(f.read_bytes()).digest()
meta={'status':'PASS','files':len(files),'bytes':a.stat().st_size,'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'verified_manifest_entries':len(checked)};(r/'runs/newtask_batch_backup_20261013.json').write_text(json.dumps(meta,indent=2));print(meta)
