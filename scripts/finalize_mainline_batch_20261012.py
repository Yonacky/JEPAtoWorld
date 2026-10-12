from pathlib import Path
import hashlib,json,re,zipfile
r=Path('F:/2610/CVPR27');checked=[]
for name,mf in [('mainline_history_20261012','artifact_manifest.json'),('mainline_fourway_20261012','artifact_manifest.json'),('mainline_regroup_20261012','manifest.json')]:
    folder=r/'runs'/name; z=json.loads((folder/mf).read_text(encoding='utf-8-sig'))
    for key,val in z.items():
        expected=val['sha256'] if isinstance(val,dict) else val
        p=(r/key) if key.replace(chr(92),'/').split('/')[0] in ['runs','scripts','docs'] else folder/key
        assert p.exists(),str(p)
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,str(p)
        checked.append({'path':str(p.relative_to(r)),'sha256':expected})
p=r/'docs/MASTER_PLAN.md';s=p.read_text(encoding='utf-8-sig')
start=s.index('当前定位：');end=s.index('\n\n',start)
s=s[:start]+'当前定位：现已并行推进R2-2局部四格、R2-3无标签跨块重组、R2-5历史对照，以及R2-6干预编辑诊断。A-3、B-3、H-1/H-2、N-3本轮均完成并核验；结果仍是旧数据探索，没有把任何一条主线标为全部完成。两参照静态编辑基线保留，外推失败也保留。已保存计划是六步总路线与四步干预增补，不使用未经核实的七阶段编号。'+s[end:]
lines=s.splitlines()
for i,line in enumerate(lines):
    if line.startswith('| R2-2 |'):lines[i]='| R2-2 | 路线A：全局非正交与局部坐标 | 部分完成，A-3最简四格已核验 | [四格报告](mainline_fourway_report_20261012.txt)；未胜原父块，保留路由信息与容量限制 | 连续坐标对齐、预算匹配及独立片段仍待验证，不盲目扩容 |'
    if line.startswith('| R2-3 |'):lines[i]='| R2-3 | 路线B：公共与差异成分 | 部分完成，B-3重组已核验 | [重组报告](mainline_regroup_report_20261012.txt)；局部正信号不稳定，未恢复普遍紧凑差异 | 先形成新机制依据，再做独立片段确认 |'
    if line.startswith('| R2-5 |'):lines[i]='| R2-5 | 历史与动力学解释 | H-1最简对照及H-2来源审计完成，系统验证待做 | [历史报告](mainline_history_report_20261012.txt)；[边界审计](history_boundary_audit_20261012.txt) | 实际episode/reset语义、预测导向压缩、物理干预与新轨迹复现 |'
s='\n'.join(lines)+'\n'
# All active lower-level tables precede historical records, preserving their IDs and contents.
parts=re.split(r'(?m)(?=^## )',s);history=next(i for i,t in enumerate(parts) if t.startswith('## 已完成批次记录'));late=[]
for i in range(len(parts)-1,history,-1):
    if parts[i].startswith('## ') and any(code in parts[i].split('\n\n',1)[0] for code in ['路线A','四格']):late.insert(0,parts.pop(i))
if late:parts[history:history]=late
s=''.join(parts)
newrows='''| 2026-10-12 | A-3局部与非正交四格 | 已完成；3seed，132不变量检查、372预测复算；未胜原父块 | 不以稀疏重构下降宣称因素解释改善 | [报告](mainline_fourway_report_20261012.txt) |
| 2026-10-12 | B-3无标签多滞后跨块重组 | 已完成；84配置、504目标复算，差异收益不稳定 | 保留局部正信号与退化，不扩大模型 | [报告](mainline_regroup_report_20261012.txt) |
| 2026-10-12 | H-1历史对照及H-2边界来源 | 已完成；全历史读出改善，同维压缩及未来预测无一致收益 | 先核对实际序列生成与重采样语义，再形成独立动力学证据 | [历史报告](mainline_history_report_20261012.txt)；[边界审计](history_boundary_audit_20261012.txt) |
| 2026-10-12 | N-3编辑目标诊断 | 已完成；目标遗漏与投影干扰精确分解，独立数值核验通过 | 目标保留、非目标泄漏与恢复率同时约束 | [报告](edit_objective_report_20261012.txt) |
'''
s=s.replace('## 维护与协作规则',newrows+'\n## 维护与协作规则')
s+='\n| 2026-10-12 | 根智能体合并四主线结果与证据 | 用户要求不局限N分支 | 各主线均保持部分完成；未新增确认、未扩容、未推送 |\n'
s=re.sub(r'(?m)(^\|[^\n]*\n)(?:[ \t]*\n)+(?=\|)',r'\1',s)
p.write_text(s,encoding='utf-8')
for target in re.findall(r'\]\(([^)]+)\)',s):
    if '://' not in target and not target.startswith('#'):assert (p.parent/target).resolve().exists(),target
report='''多主线探索批次汇总 2026-10-12
本轮完成A-3、B-3、H-1/H-2和N-3。三个Astra子智能体负责A/B/历史，主智能体负责编辑诊断及交叉审阅。全部为现有缓存上的CPU计算，无新增GPU占用或编码器训练；没有使用旧开发集冒充独立确认。

A-3：局部/全局与正交/非正交四格，4因素3种子。局部模型的单坐标及完整4坐标读出均值均低于原父块。Pong ball_y分组编号本身R²=.662，高于局部正交.641和非正交.631。仅已知gate时局部逆变换可恢复输入，单独4维z未证明全局单射。不得把重构降低、更多参数或门控语义写成解释轴成功。
B-3：多滞后相关的无标签重组，4/8维候选数一致，标签只用于诊断选组及读出。Pong4维差异保持.793；Pinball4维首个差异由.399降到.244。8维第二差异有局部改善，但其他因素受损且重采样不稳定。未支持普遍紧凑恢复。
H-1：4帧全历史动力学读出平均R²，Pong .040->.159，Pinball .173->.330；full512维相对单帧/复制秩128有容量差。压到128维后Pong .068增益不确定、Pinball .104低于单帧；未来1/2/4步表示预测无一致改善。H-2找到Pinball生成器重采样代码，但未与实际数据生成版本/settings绑定；缓存没有episode/reset不等于原始数据没有。只支持连续索引窗口探索，不能当完整轨迹或物理因果证据。
N-3：正交编辑误差恰等于目标遗漏+投影干扰。原两参照joint0颜色误差.732=.283遗漏+.449干扰。更强响应投影保留目标更多却误差2.666；强抑制误差降但颜色恢复下降。当前后处理目标与解释功能之间存在实测取舍。

核验：A 372预测重算最大差1.42e-14；B 84配置/504目标最大预测差1.38e-14；H独立直接解最大差2.36e-13；N全空间最大差5.25e-13。根智能体另核对三条agent产物清单与总计划证据链接。细节和限制见各分支报告，不将这次实现核验外推为泛化成立。

后续优先级：历史线先闭合原始数据边界，再比较同维预测导向压缩与PCA；视觉解释线保留原父块和两参照基线，下一方法需同时保留目标、控制干扰，避免单纯加大稀疏性或局部模型数量。A/B暂不沿本次失败目标盲目扩容，未见物体/视角和真实物理参数干预仍在总计划中，不以本轮代替。
'''
(r/'docs/mainline_batch_report_20261012.txt').write_text(report,encoding='utf-8')
(r/'runs/mainline_batch_validation_20261012.json').write_text(json.dumps({'status':'PASS','files_verified':len(checked),'files':checked,'plan_links':'PASS'},indent=2))
print('PASS',len(checked),'artifact hashes; plan links valid')

