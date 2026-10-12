"""G-1 fixed-route LIBERO temporal transfer diagnostic, CPU, demo-disjoint."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import numpy as np
from scipy import linalg as la
import hashlib,json,argparse
R=Path('F:/2610/CVPR27')
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=R/'runs/complex_pilot_20261011/libero');ap.add_argument('--output',type=Path,default=R/'runs/complex_history_transfer_20261012');ap.add_argument('--report',type=Path,default=R/'docs/complex_history_transfer_report_20261012.txt');ap.add_argument('--dataset-name',default='LIBERO bowl-on-plate');ap.add_argument('--data-status',default='Previously inspected dev, exploratory');args=ap.parse_args();I=args.input;O=args.output;O.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
A=np.array([1e-6,1e-4,.01,.1,1,10,100]);z=np.load(I/'sample_manifest.npz');x=np.load(I/'features.npz')['mean'].astype(float);y=z['labels'];ep=z['episode'];sp=z['split'];fr=z['frame'];le=z['episode_length'];D=64
sets=[set(ep[sp==s]) for s in range(3)];assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
indices=[]
for s in range(3):
 rows=[]
 for demo in sorted(sets[s]):
  q=np.flatnonzero(ep==demo);assert np.all(sp[q]==s);assert np.all(np.diff(fr[q])==4);assert len(np.unique(le[q]))==1
  for j in range(3,len(q)-2):rows.append(q[j-3:j+3])
 indices.append(np.array(rows))
assert all(v.shape[1]==6 for v in indices)
xc=x[sp==0].mean(0);u,sv,vh=la.svd(x[sp==0]-xc,full_matrices=False);base=vh[:D].T/(sv[:D]/np.sqrt((sp==0).sum()));f=(x-xc)@base
H=[np.concatenate([f[q[:,3-j]] for j in range(4)],axis=1)/2 for q in indices];hm=H[0].mean(0);HH=[v-hm for v in H];F=[np.concatenate([f[q[:,4]],f[q[:,5]]],axis=1) for q in indices]
Y=[np.column_stack([y[q[:,3]],y[q[:,3]]-y[q[:,2]],ff,hh]) for q,ff,hh in zip(indices,F,H)];ym=Y[0].mean(0);ys=Y[0].std(0);assert np.all(ys>1e-8);YY=[(v-ym)/ys for v in Y]
groups={'position':slice(0,3),'gripper':slice(3,5),'backward_displacement':slice(5,8),'gripper_change':slice(8,10),'future4frames':slice(10,74),'future8frames':slice(74,138),'history_reconstruction':slice(138,394)}
G=HH[0].T@HH[0]/len(HH[0]);C=HH[0].T@(F[0]-F[0].mean(0))/len(HH[0]);ev,pca=la.eigh(G);B=la.solve(G+.01*np.eye(256),C,assume_a='pos');ee,v=la.eigh(C.T@B);pb,_=la.qr(B@v[:,-D:],mode='economic')
progress=[fr[q[:,3]]/np.maximum(le[q[:,3]]-1,1) for q in indices]
methods={'current64':[f[q[:,3]] for q in indices],'history256':H,'history_pca64':[v@pca[:,-D:] for v in HH],'history_rrr64':[v@pb for v in HH],'progress_linear':[v[:,None] for v in progress],'progress_cubic':[np.column_stack([v,v*v,v*v*v]) for v in progress]}
cfg={'scope':args.data_status,'dataset':args.dataset_name,'device':'CPU','code_sha256':sha(Path(__file__)),'sources':{str(p):sha(p) for p in [I/'features.npz',I/'sample_manifest.npz',I/'features.json']+([I/'config.json'] if (I/'config.json').exists() else [I/'data.json',I.parent/'config.json'])},'fixed_budget':{'per_frame':64,'history':4,'full_history_width':256,'compression':64,'predictive_ridge':.01,'future_training_target_width':128,'future_steps_original_frames':[4,8],'readout_alphas':A.tolist()},'protocol':'Base whitenedPCA64 train frames only; historyPCA and RRR fit train windows only. RRR lambda fixed from prior route, no new search. Readout regularization tune only independently per target group. All routes common crop.','window_counts':[len(q) for q in indices],'demos':[sorted(v) for v in sets],'targets':'obs ee_pos xyz and2 raw gripper joints; backward displacement=current-minus-previous sampled obs, delta4 original frames not velocity. Future=retained JEPA features, not physical state.','progress':'Offline shortcut using full demo length, not deployable observation','statistical_unit':str(len(sets[2]))+' dev demos,1000 paired demo bootstrap; too few for broad generalization','capacity':'PCA/RRR64 matched readout width, differing objectives; fullhistory256 larger capacity. Static frozen JEPA encoder, no actions or physics interventions.'}
dump(O/'config.json',cfg);np.savez_compressed(O/'transforms_indices_truth.npz',base_center=xc,base=base,pca_history=pca[:,-D:],rrr_history=pb,history_center=hm,target_mean=ym,target_std=ys,truth=YY[2],**{s:q for s,q in zip(['train','tune','dev'],indices)},dev_episode=ep[indices[2][:,3]],dev_frame=fr[indices[2][:,3]])
results=[];checks=[];preds={};demo=ep[indices[2][:,3]]
for name,V in methods.items():
 vm=V[0].mean(0);VV=[v-vm for v in V];g=VV[0].T@VV[0]/len(VV[0]);c=VV[0].T@YY[0]/len(VV[0]);e,v=la.eigh(g);e=np.maximum(e,0);rhs=v.T@c;weights=[];loss=[]
 for a in A:
  w=v@(rhs/(e[:,None]+a));weights.append(w);t=VV[1]@w;loss.append(np.mean((t-YY[1])**2,axis=0))
 loss=np.asarray(loss);choice=np.zeros(394,dtype=int)
 for sl in groups.values():choice[sl]=np.argmin(loss[:,sl].mean(1))
 w=np.column_stack([weights[choice[j]][:,j] for j in range(394)]);p=VV[2]@w;preds[name]=p;err=0.
 for ai in np.unique(choice):
  cols=np.flatnonzero(choice==ai);ww=la.solve(g+A[ai]*np.eye(len(g)),c[:,cols],assume_a='pos');err=max(err,float(np.max(np.abs(VV[2]@ww-p[:,cols]))))
 assert err<1e-5;np.savez_compressed(O/f'{name}.npz',prediction=p,weights=w,center=vm,alpha=A[choice],tune_mse=loss)
 pp=np.load(O/f'{name}.npz')['prediction'];assert np.allclose(pp,p,atol=1e-12)
 checks.append({'method':name,'direct_prediction_error':err,'feature_rank':int(np.sum(e>max(e.max()*1e-9,1e-12))),'width':len(g)})
 for group,sl in groups.items():
  den=((YY[2][:,sl]-YY[2][:,sl].mean(0))**2).sum(0);r2=1-((p[:,sl]-YY[2][:,sl])**2).sum(0)/np.maximum(den,1e-12);ds=[float(np.mean((p[demo==d,sl]-YY[2][demo==d,sl])**2)) for d in sorted(sets[2])]
  results.append({'method':name,'target':group,'r2_components':[float(v) if dd>1e-12 else None for v,dd in zip(r2,den)],'macro_demo_standardized_mse':float(np.mean(ds)),'demo_mse':dict(zip(sorted(sets[2]),ds)),'alpha':float(A[choice[sl.start]])})
 print(name,[(r['target'],round(r['macro_demo_standardized_mse'],4)) for r in results if r['method']==name],flush=True)
# Future persistence directly copies current retained feature, with no fitted parameter.
persistence=np.column_stack([f[indices[2][:,3]],f[indices[2][:,3]]]);pp=(persistence-ym[10:138])/ys[10:138]
for target,sl in [('future4frames',slice(0,64)),('future8frames',slice(64,128))]:
 ds=[float(np.mean((pp[demo==d,sl]-YY[2][demo==d,10:138][:,sl])**2)) for d in sorted(sets[2])];results.append({'method':'future_persistence','target':target,'macro_demo_standardized_mse':float(np.mean(ds)),'demo_mse':dict(zip(sorted(sets[2]),ds))})
comparisons=[];rng=np.random.default_rng(20261012);boots=rng.integers(len(sets[2]),size=(1000,len(sets[2])))
for name,ref in [('history256','current64'),('history_pca64','current64'),('history_rrr64','current64'),('history_rrr64','history_pca64')]:
 for group in groups:
  a=next(r for r in results if r['method']==name and r['target']==group);b=next(r for r in results if r['method']==ref and r['target']==group);delta=np.array(list(b['demo_mse'].values()))-np.array(list(a['demo_mse'].values()));comparisons.append({'method':name,'reference':ref,'target':group,'positive_is_improvement_mean':float(delta.mean()),'demo_bootstrap_ci95':np.percentile(delta[boots].mean(1),[2.5,97.5]).tolist()})
dump(O/'results.json',{'rows':results,'comparisons':comparisons});dump(O/'validation.json',{'status':'passed','checks':checks,'demo_disjoint':True,'within_demo_stride4_windows':True,'saved_metrics_recomputed':True})
lines=['G-1 LIBERO真实demo分组的历史路线外推；2026-10-12。',f'数据：{args.dataset_name}；{len(x)}帧冻结JEPA1280维FP32；完整demo划分{list(map(len,sets))}；数据身份：{args.data_status}。','四帧历史间隔4个原始帧；共同裁剪后train/tune/dev='+str([len(q) for q in indices])+'；每个窗口含前三帧、当前及后两采样帧，逐一检查同demo同split和stride4。','固定每帧train白化PCA64，历史256；比较当前64、全历史256、历史PCA64、预测导向RRR64（lambda固定0.01），无候选微调。RRR拟合目标是未来4/8原始帧的表示拼接128维，仅train；它是任务对齐自监督，不是无监督因素恢复。','读出位置xyz、两个原始夹爪关节、后向4帧位置位移/夹爪变化、未来表示、历史重构；没有真实速度/动作/质量/摩擦。位移不是速度，不假定采样dt。读出alpha只tune按组选择。',f'以下指标为{len(sets[2])}个dev demo宏平均、按train各维标准差归一的MSE（越低越好）；逐轴R2和逐demo值保存在results.json。进度线性/三次是使用完整demo长度的离线捷径诊断。']
for name in list(methods)+['future_persistence']:
 lines.append('\n'+name)
 for r in results:
  if r['method']==name:lines.append(r['target']+': '+f"{r['macro_demo_standardized_mse']:.6f}")
lines.append('\n同demo配对差：参照MSE减方法MSE，正值改善；1000次demo bootstrap，dev示范数量很少，区间不能支持广泛泛化。')
for c in comparisons:lines.append(f"{c['method']} vs {c['reference']} {c['target']}: {c['positive_is_improvement_mean']:.6f}, CI={c['demo_bootstrap_ci95']}")
lines+=['\n全部路线独立直接线性求解重算通过，未使用远端GPU或生成新特征。该批是从合成小场景方法到复杂机器人场景的路线级探索，仍非跨任务或真实物理参数验证。',f'证据：{O}；配置、源哈希、demo/帧索引、PCA/RRR映射、预测/权重/超参数与独立核验均保存。']
args.report.write_text('\n'.join(lines),encoding='utf8');print('DONE',sum(p.stat().st_size for p in O.iterdir()),flush=True)
