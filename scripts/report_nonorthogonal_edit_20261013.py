from pathlib import Path
import json,numpy as np
O=Path('runs/nonorthogonal_edit_20261013');r=json.loads((O/'results.json').read_text());inv=json.loads((O/'invariants.json').read_text());lines=['G-7g 有界非正交分块编辑报告（2026-10-13）','','执行与结论分开：四次固定200步拟合完成，没有增加配置。tune误差+泄漏在两族均选择λ1。主比较是同目标同预算oblique_lambda1对orth_lambda1，joint0双参照为原方法参照。旧ID32/OOD32已被查看，本轮属于探索，不能称独立确认。','','方法：固定旧train32学习PCA64和joint0初值，tune8仅选择λ0/1。三因素position/color/background各rank16及补块rank16，P=W E W^-1。正交族奇异值固定1，非正交族限制[1/sqrt2,sqrt2]；两族使用相同200步和30次Armijo回溯预算。目标是场景等权的原5120维相对编辑MSE加其他因素单独响应泄漏。每个scene与半程偏移scene双向配对；两参照000/222校准场景offset，匹配完整27状态。使用因素身份和真实干预目标训练，不是无监督发现。','','所有固定候选：表中误差为三因素等权，泄漏为三因素平均。']
for dom in r:
 lines+=['',dom+' | 模型 | 全空间相对误差 | 跨因素泄漏 | tune选择分数（所有域均计算但仅tune用于选择）']
 for n,v in r[dom]['models'].items():lines.append(f"{dom} | {n} | {v['mean_error']:.8f} | {np.mean(v['cross_factor_leakage']):.8f} | {v['selection_score']:.8f}")
lines+=['','逐因素全部指标（顺序：原空间相对误差、目标投影能量比、目标最近邻正确率、两个非目标均保持）。能量比不是真实目标恢复率；斜投影不使用正交Pythagorean分解，且可能放大能量。']
for dom in r:
 for n,v in r[dom]['models'].items():
  for k,f in enumerate(['position','color','background']):lines.append(dom+' | '+n+' | '+f+' | '+' | '.join(f'{q:.8f}' for q in v['factor_means'][k]))
lines+=['','几何与噪声：所有四块均rank16，幂等、两两乘积零、四块和I误差小于1e-14。非正交块不对称，因此不能称欧氏正交。完整变换保留PCA64内信息，不恢复PCA已舍弃信息；可逆性也不保证单块编辑语义正确。噪声增益按PCA64等方差输入噪声定义，均方增益=||P||F²/64，单位谱范数给出最坏方向放大。']
for n,v in inv.items():lines.append(n+' | cond='+str(v['condition'])+' | singular[min,max]='+str([v['singular_min'],v['singular_max']])+' | block_norms='+str(v['projector_spectral_norm'])+' | noise_gains='+str(v['isotropic_noise_mean_squared_gain']))
lines+=['','解读：非正交λ1相对正交λ1，ID平均误差.44285→.40778，OOD .70073→.63329；颜色目标准确率ID .68967→.80230，OOD .45385→.52293。OOD背景两个非目标保持.80100→.88433，仍未完全保持。相对joint0双参照，ID颜色投影能量比.71503→.67168下降，但实际编辑误差.74288→.57525、正确率.71955→.80230改善。仅优化目标误差不能用投影能量比解释为完整因素恢复。','','边界与后续判断：保留此路线作为有监督可编辑分块的候选，不宣称盲源分离成功。两个非正交解均靠近cond=2边界，最坏单块谱范数约1.226，显示有代价的斜投影；本轮不扩大边界。只有一个初始化和固定200步，优化预算公平不等于全局最优或等有效自由度（非正交族更丰富）。原编码器、静态三因素和两参照干预设置的收益不自动推广到物理因素或其他编码器。旧A3失败没有被推翻，任务和监督不同；N4的retention条件也没有满足声明。出处与实质差异见docs/nonorthogonal_edit_sources_20261013.txt。']
p=O/'independent_validation.json'
if p.exists():
 a=json.loads(p.read_text());lines+=['','独立核验：不导入拟合脚本/旧construct；Python枚举27状态重建配对，独立坐标掩码再解线性系统，逐编辑直接原5120维误差与27候选距离复算，核对全部场景指标、泄漏、tune选择与源hash。']
 for d,v in a['domains'].items():lines.append(d+' '+str(v))
 lines+=['','探索配对95% CI：每个域16个独立scene-pair，双向先平均，固定seed20261013、5000次bootstrap；无多重比较校正，区间不含训练拟合或模型选择不确定性。差值=非正交λ1减参照，误差为负有利。']
 for n,v in a['paired_bootstrap'].items():
  lines.append(n+' overall error '+str(v['overall_error_difference'])+' CI '+str(v['overall_error_ci95']))
  for k,f in enumerate(['position','color','background']):lines.append(f+' metric deltas '+str(v['factor_means'][k])+' lower '+str(v['factor_ci95'][0][k])+' upper '+str(v['factor_ci95'][1][k]))
else:lines+=['','独立全空间复算尚在运行；此报告未最终归档。']
Path('docs/nonorthogonal_edit_report_20261013.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
