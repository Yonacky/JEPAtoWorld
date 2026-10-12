from pathlib import Path
import re,json,hashlib,zipfile
r=Path('F:/2610/CVPR27')
report='''路线级复杂数据研究批次，2026-10-12
本轮完成G-1/G-2/G-3与根协调G-0。按用户要求扩大粒度，不继续调N-4颜色权重。服务器恢复后，新增LIBERO炉台取碗716帧、抽屉取碗744帧，共40条demo，各任务12/4/4互斥划分，单张授权H100完成冻结I-JEPA FP32提取。新任务分别训练评价，并非零样本跨任务泛化。所有方法计算本地CPU，结果已保存F盘，无GitHub推送。

路线判断一：保留简单白化和SFA作为主要基线，现有ICA/分块不继续扩容。原LIBERO、CausalWorld mean/grid四代表路线比较，ICA三种子均收敛但无稳定优势。新增两任务静态4坐标位置读出R²：炉台white .875、SFA .930；抽屉white .859、SFA .875。夹爪收益不一致。坐标由train标签诊断选择，SFA变换拟合本身无标签，因此这些结果不等于自动发现语义因素。完整64维正交变换同span预测一致，不是产生新信息；原分块局部收益仍保留。不同报告选择预算/归一化/crop不同，不直接合并数值排名。

路线判断二：优先输入和压缩的信息保留。CausalWorld物体位置的demo等权标准化MSE，完整64维mean .839、grid .712，raw线性参照 .815/.617；但grid物体绿色64维1.013、raw .518。空间信息帮助位置、低秩压缩损失颜色的格局比多数同span旋转差异更突出。grid5120与mean1280输入容量不同，raw有限样本ridge也不是信息论上界，不能声称纯方法优势或证明全部差异只来自PCA。

路线判断三：保留历史对状态变化的诊断作用，暂停此RRR未来压缩支线微调。三个LIBERO任务冻结协议均未让history256/PCA64/RRR64超过当前64的未来4/8帧表示预测。完整历史部分改善夹爪/后向位移；新任务位移MSE当前/历史：炉台.304/.258，抽屉.221/.198，其中炉台小收益区间跨零。RRR未稳定胜PCA。未来表示预测不是实际物理状态预测，也没有动作输入，不能据此否定动作条件世界模型。任务进度捷径仍需控制。

证据规模与边界：G2共456预测及360坐标/组选取复算通过；G3两任务100预测从原mean特征和保存参数重算最大差2.22e-16；G1三个任务独立线性求解误差≤7.29e-13。根另核验525项文件哈希、两新任务的demo互斥、stride和逐demo位移一致性。每新task只有4个dev demo，旧任务及CW序列已经使用过；均属开发探索，不能宣称物理因素独立变化或广泛泛化。

下一轮按三个大包推进，不展开当前失败目标的小修小补：G-4 信息保留包，固定mean/grid和未压缩参照，比较能保留物体位置、颜色的表示预算；G-5 动力学包，在可恢复状态的仿真中先核对动作/观测时间，再做同初态动作分支及物理参数干预；G-6 解释功能包，把保留下来的表示与white/SFA/原块放入真实单因素/组合干预评价，要求目标正确与非目标保持同时改善。三个包分别回答输入是否足够、动态是否可辨、编辑是否有效，仍隶属原R2路线，不重编阶段。
'''
(r/'docs/route_level_batch_report_20261012.txt').write_text(report,encoding='utf-8')
p=r/'docs/MASTER_PLAN.md';s=p.read_text(encoding='utf-8-sig');a=s.index('当前定位：');b=s.index('\n\n',a);s=s[:a]+'当前定位：G-0至G-3路线级复杂数据批次已完成并核验，新增两个LIBERO任务与三任务历史对照、CausalWorld代表方法比较。保留white/SFA与历史状态变化诊断，暂不扩容当前ICA/分块或微调RRR未来压缩。下一步转向输入保留、动作/物理干预与解释功能三个实验包；整体R2计划仍部分完成。'+s[b:]
lines=s.splitlines()
for i,l in enumerate(lines):
 if l.startswith('| G-0 |'):lines[i]='| G-0 | R2-4 / R2-5 / R2-6 | 多路线统一收口与新增任务数据协调 | 已完成并核验备份；2026-10-12 | 主智能体 | 单H100新增1460帧、两任务demo互斥；525文件哈希通过；[汇总](route_level_batch_report_20261012.txt) |'
 if l.startswith('| R2-4 |'):lines[i]='| R2-4 | 合并检查并冻结候选 | 路线级收敛首批完成，整体部分完成 | [G批次汇总](route_level_batch_report_20261012.txt)；white/SFA保留，当前ICA/分块暂不扩容 | 输入保留、受控动力学及解释功能包后再冻结论文方法 |'
 if l.startswith('| R2-5 |'):lines[i]='| R2-5 | 历史与动力学解释 | H系列与G-1三任务探索完成，物理验证待做 | [三任务历史](complex_history_transfer_three_tasks_20261012.txt)；部分变化读出改善，未来表示预测未改善 | G-5动作时间对齐与同初态物理分支，不能用静态/历史读出替代 |'
 if l.startswith('| R2-3 |'):lines[i]='| R2-3 | 路线B：公共与差异成分 | B系列及G-2代表路线完成探索 | [复杂路线报告](complex_route_comparison_report_20261012.txt)；保留原块局部信号、暂停当前目标扩容 | G-4输入保留及G-6真实干预功能验证 |'
s='\n'.join(lines)+'\n';new='''## 下一轮大粒度实验包

| 编号 | 所属 | 实验包 | 状态 | 验收条件 |
|---|---|---|---|---|
| G-4 | R2-1 / R2-3 | 空间输入与压缩信息保留 | 待执行 | 同源mean/grid与原高维参照，位置/颜色分别评价，不把容量差当方法收益 |
| G-5 | R2-5 / R2-6 | 同初态动作与物理因素分支 | 待执行 | 先核对动作观测对齐与状态恢复，再固定动作及物理干预；记录真实边界 |
| G-6 | R2-2 / R2-6 | 候选表示的单因素及组合解释功能 | 待执行 | white/SFA/原块必要对照，目标恢复与非目标保持共同判断，分开监督等级 |

''';s=s.replace('## 已完成批次记录',new+'## 已完成批次记录');s=s.replace('## 维护与协作规则','| 2026-10-12 | G-0至G-3复杂数据路线级批次 | 已完成；两新任务1460帧、三任务历史、CW输入和方法对照 | 保留简单基线，转入信息保留与真实动力学/干预包 | [报告](route_level_batch_report_20261012.txt) |\n\n## 维护与协作规则');s+='\n| 2026-10-12 | G系列首批收尾并登记G-4/5/6实验包 | 用户要求大粒度研究；跨任务未复现RRR预测优势、输入瓶颈突出 | 失败记录保留，原计划层级不变，未推送 |\n';s=re.sub(r'(?m)(^\|[^\n]*\n)(?:[ \t]*\n)+(?=\|)',r'\1',s);p.write_text(s,encoding='utf-8')
for t in re.findall(r'\]\(([^)]+)\)',s):
 if '://' not in t and not t.startswith('#'):assert (p.parent/t).resolve().exists(),t
folders=['complex_history_transfer_20261012','complex_history_transfer_stove_20261012','complex_history_transfer_drawer_20261012','complex_route_comparison_20261012','libero_multitask_20261012','libero_multitask_preparation_20261012'];files=set()
for n in folders:files.update(f for f in (r/'runs'/n).rglob('*') if f.is_file() and f.suffix!='.zip')
for folder,patterns in [('docs',['*route*20261012*','*history_transfer*','libero_multitask*','MASTER_PLAN.md']),('scripts',['*route*20261012*','complex_history_transfer*','*libero*20261012*']),('runs',['route_batch_root_validation*','libero_multitask_*20261012.json'])]:
 for pat in patterns:files.update(f for f in (r/folder).glob(pat) if f.is_file())
a=r/'experiments/route_level_batch_20261012.zip';assert not a.exists()
with zipfile.ZipFile(a,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in sorted(files):z.write(f,f.relative_to(r).as_posix())
with zipfile.ZipFile(a) as z:
 assert z.testzip() is None
 for f in files:assert hashlib.sha256(z.read(f.relative_to(r).as_posix())).digest()==hashlib.sha256(f.read_bytes()).digest()
meta={'status':'PASS','files':len(files),'bytes':a.stat().st_size,'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'plan_links':'PASS'};(r/'runs/route_level_batch_backup_20261012.json').write_text(json.dumps(meta,indent=2));print(meta)
