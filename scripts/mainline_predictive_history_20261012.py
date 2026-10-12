"""H-3 CPU predictive reduced-rank history compression; old-r0 exploration."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import numpy as np
from scipy import linalg as la
import json,hashlib,platform
R=Path('F:/2610/CVPR27');O=R/'runs/mainline_predictive_history_20261012';O.mkdir(exist_ok=True)
A=np.array([1e-6,1e-4,.01,.1,1,10,100]); L=[.01,1.,100.];SPLITS=['train','tune','dev']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def scores(y,p):return 1-((y-p)**2).sum(0)/np.maximum(((y-y.mean(0))**2).sum(0),1e-14)
def ci(y,p,q):
 rng=np.random.default_rng(20261012);blocks=[np.arange(i,min(i+64,len(y))) for i in range(0,len(y),64)];v=[]
 for _ in range(200):
  ix=np.concatenate([blocks[i] for i in rng.integers(len(blocks),size=len(blocks))]);v.append(float(np.mean(scores(y[ix],p[ix])-scores(y[ix],q[ix]))))
 return np.percentile(v,[2.5,97.5]).tolist()
cfg={'scope':'Old r0 exploratory; unknown episode/reset identities. Same continuous-index windows as H-1.','cpu':True,'numpy':np.__version__,'python':platform.python_version(),'code_sha256':sha(Path(__file__)),'rank':[64,128],'transform_lambdas':L,'readout_alphas':A.tolist(),'targets':'Train next1,next2,next4 retained whitened128 concatenation, total384','objective':'min_W ||H W-F||^2/n + lambda||W||^2, rank(W)<=r. B=(G+lambda I)^-1 C V_r; V_r eigenvectors of C.T (G+lambda I)^-1 C. QR(B) gives orthonormal coordinates of same predictive input subspace.','selection':'For each rank choose lambda solely by tune MSE of direct reduced-rank prediction before downstream readout; report all candidates. No dynamics/dev selection.','readout':'Train linear ridge; tune separately per dynamics component and per horizon; reconstruction uses one alpha for all512 history coordinates','capacity':'RRR transform512xr and train384-dimensional future task supervision; PCA512xr uses variance only. Same downstream width for equal rank, not equal training objectives or compute.','retention':'Ridge reconstruction of full512 history at dev; train mean error normalized MSE plus coordinate-average R2','bootstrap':'Paired64-frame blocks200; conditional on reused segment, not independent trajectories','limitations':'Task-aligned self-supervised prediction learning, not unsupervised factor recovery; no action input/physics intervention; unknown reset.'}
dump(O/'config.json',cfg); rows=[];checks=[];audits=[];cmp=[]
for ds,ids in [('pong',[0,1]),('pinball',[1,3])]:
 cp=R/f'runs/e04_inputs/cache/{ds}_ijepa_r0.npz';z=np.load(cp);X=[z[s] for s in SPLITS];S=[z['source_'+s] for s in SPLITS];ix=[]
 assert all(np.all(np.diff(s)>0) for s in S)
 assert all(not np.intersect1d(S[i],S[j]).size for i in range(3) for j in range(i))
 for s in S:
  t=np.arange(3,len(s)-4);ix.append(t[np.all(np.stack([s[t+k]==s[t]+k for k in range(-3,5)]),axis=0)])
 source=[s[t] for s,t in zip(S,ix)];assert all(np.all(np.diff(s)==1) for s in source)
 H=[np.concatenate([x[t-j] for j in range(4)],axis=1)/2 for x,t in zip(X,ix)];hm=H[0].mean(0);HH=[h-hm for h in H]
 F=[np.concatenate([x[t+j] for j in [1,2,4]],axis=1) for x,t in zip(X,ix)];fm=F[0].mean(0)
 Y=[np.column_stack([z['target_'+s][t][:,ids],f,h]) for s,t,f,h in zip(SPLITS,ix,F,H)];ym=Y[0].mean(0)
 np.savez_compressed(O/f'{ds}_truth_indices.npz',truth=Y[2][:,:386].astype('float32'),**{s:v for s,v in zip(SPLITS,source)},**{'rows_'+s:v for s,v in zip(SPLITS,ix)})
 G=HH[0].T@HH[0]/len(HH[0]);C=HH[0].T@(F[0]-fm)/len(HH[0]);ev,pca=la.eigh(G)
 methods=[('current128',[x[t] for x,t in zip(X,ix)],None,{}),('history512',H,np.eye(512),{})]
 for rank in [64,128]:methods.append((f'pca{rank}',[h@pca[:,-rank:] for h in HH],pca[:,-rank:],{}))
 selection={};cand=[]
 for lam in L:
  Bfull=la.solve(G+lam*np.eye(512),C,assume_a='pos');e,v=la.eigh(C.T@Bfull)
  for rank in [64,128]:
   vr=v[:,-rank:];B=Bfull@vr;q,_=la.qr(B,mode='economic');pr=HH[1]@B@vr.T+fm;loss=float(np.mean((pr-F[1])**2));name=f'rrr{rank}_lambda{lam:g}'
   eigerr=float(np.max(np.abs((G+lam*np.eye(512))@B-C@vr)))
   info={'lambda':lam,'rank':rank,'transform_tune_mse':loss,'normal_equation_max_error':eigerr,'orthogonality_error':float(np.max(np.abs(q.T@q-np.eye(rank)))),'B_rank':int(np.linalg.matrix_rank(B))}
   cand.append((rank,loss,name));methods.append((name,[h@q for h in HH],q,info))
   np.savez_compressed(O/f'{ds}_{name}_transform.npz',basis=q,B=B,future_basis=vr,history_center=hm,future_center=fm)
 for rank in [64,128]:selection[str(rank)]=min((loss,name) for r,loss,name in cand if r==rank)[1]
 audits.append({'dataset':ds,'cache_sha256':sha(cp),'schema_sha256':sha(R/f'experiments/e03_fp32_20261011/{ds}_factor_schema.json'),'source_hashes':{s:hashlib.sha256(v.tobytes()).hexdigest() for s,v in zip(SPLITS,S)},'counts':list(map(len,ix)),'selected_by_tune_future':selection})
 preds={}
 for name,V,basis,info in methods:
  vm=V[0].mean(0);VV=[v-vm for v in V];g=VV[0].T@VV[0]/len(VV[0]);c=VV[0].T@(Y[0]-ym)/len(VV[0]);e,v=la.eigh(g);e=np.maximum(e,0);rhs=v.T@c
  losses=[];ws=[]
  for a in A:
   w=v@(rhs/(e[:,None]+a));ws.append(w);losses.append(np.mean((VV[1]@w+ym-Y[1])**2,axis=0))
  losses=np.asarray(losses);choice=np.zeros(898,dtype=int)
  for sl in [slice(0,1),slice(1,2),slice(2,130),slice(130,258),slice(258,386),slice(386,898)]:choice[sl]=np.argmin(losses[:,sl].mean(1))
  w=np.column_stack([ws[choice[j]][:,j] for j in range(898)]);p=VV[2]@w+ym;preds[name]=p[:,:386];maxerr=0.
  for ai in np.unique(choice):
   cols=np.flatnonzero(choice==ai);ww=la.solve(g+A[ai]*np.eye(len(g)),c[:,cols],assume_a='pos');maxerr=max(maxerr,float(np.max(np.abs(VV[2]@ww+ym[cols]-p[:,cols]))))
  assert maxerr<1e-6,maxerr
  path=O/f'{ds}_{name}_readout.npz';np.savez_compressed(path,prediction=p[:,:386].astype('float32'),weights=w,target_center=ym,feature_center=vm,basis=np.empty((0,0)) if basis is None else basis,history_center=hm,alpha=A[choice],tune_mse=losses,chosen=choice)
  saved=np.load(path)['prediction'].astype(float);truth=np.load(O/f'{ds}_truth_indices.npz')['truth'].astype(float);err=float(np.max(np.abs(scores(truth,saved)-scores(Y[2][:,:386],p[:,:386]))));assert err<1e-5
  retention=float(np.mean((p[:,386:]-Y[2][:,386:])**2)/np.mean((Y[2][:,386:]-Y[0][:,386:].mean(0))**2))
  row={'dataset':ds,'method':name,'width':len(g),'rank':int(np.sum(e>max(e.max()*1e-9,1e-12))),'selected_transform':name in selection.values(),'dynamics_r2':scores(Y[2][:,:2],p[:,:2]).tolist(),'dynamics_mean_r2':float(scores(Y[2][:,:2],p[:,:2]).mean()),'future_mean_r2':[float(scores(Y[2][:,sl],p[:,sl]).mean()) for sl in [slice(2,130),slice(130,258),slice(258,386)]],'history_reconstruction_nmse_trainmean':retention,'history_reconstruction_mean_r2':float(scores(Y[2][:,386:],p[:,386:]).mean()),'readout_alpha_unique':np.unique(A[choice]).tolist(),**info};rows.append(row)
  checks.append({'dataset':ds,'method':name,'independent_direct_solve_max_error':maxerr,'saved_prediction_score_max_error':err});print(ds,name,'dyn',row['dynamics_mean_r2'],'future',row['future_mean_r2'],'retention',retention,flush=True)
 for rank in [64,128]:
  name=selection[str(rank)];ref=f'pca{rank}'
  for target,sl in [('dynamics',slice(0,2)),('future1',slice(2,130)),('future2',slice(130,258)),('future4',slice(258,386))]:cmp.append({'dataset':ds,'selected_rrr':name,'reference':ref,'target':target,'delta_r2':float(np.mean(scores(Y[2][:,sl],preds[name][:,sl])-scores(Y[2][:,sl],preds[ref][:,sl]))),'ci95':ci(Y[2][:,sl],preds[name][:,sl],preds[ref][:,sl])})
dump(O/'results.json',{'rows':rows,'audits':audits,'selected_vs_pca':cmp});dump(O/'validation.json',{'status':'passed','checks':checks})
lines=['H-3 预测导向历史压缩；CPU；2026-10-12。','旧r0连续索引探索，未知reset边界限制保持；train8185/tune1017/dev1017共同裁剪，不跨split。未使用动力标签拟合压缩。','四帧512维历史预测未来1/2/4帧表示拼接384维，最小化带lambda的降秩岭目标，取输入映射的QR正交基；rank64/128，每秩lambda从0.01/1/100按tune直接未来预测MSE选。所有候选均报告，不按dev或动力读出选。','这是任务对齐的自监督预测学习，未来目标与下游预测任务相同；不能称无监督因素恢复。RRR和PCA同秩匹配读出宽度，但目标统计量、计算和总拟合自由度不同。单帧128、全历史512仅参照。','保留指标：以train均值为参照的dev全512维历史重构NMSE，越低越好；重构读出只train拟合/tune正则。预测独立直接解重算及落盘评分重算通过。']
for ds in ['pong','pinball']:
 lines.append('\n'+ds+'：method / 动力平均R2 / future1,2,4平均R2 / 重构NMSE / 有效秩 / selected')
 for row in rows:
  if row['dataset']==ds:lines.append(f"{row['method']} / {row['dynamics_mean_r2']:.6f} / "+','.join(f'{x:.6f}' for x in row['future_mean_r2'])+f" / {row['history_reconstruction_nmse_trainmean']:.6f} / {row['rank']} / {row['selected_transform']}")
 for row in rows:
  if row['dataset']==ds and 'lambda' in row:lines.append(f"{row['method']} transform tune MSE={row['transform_tune_mse']:.8f}")
 for c in cmp:
  if c['dataset']==ds:lines.append(f"{c['selected_rrr']} - {c['reference']} {c['target']}: delta={c['delta_r2']:.6f}, CI={c['ci95']}")
lines+=['\n条件区间为200次64帧配对块重采样，仍条件于旧片段；不代表独立轨迹复现。未做物理参数/动作干预，本实验不完成R2-5。','证据：runs/mainline_predictive_history_20261012/config.json、results.json、validation.json、truth_indices及各变换/读出NPZ。']
(R/'docs/mainline_predictive_history_report_20261012.txt').write_text('\n'.join(lines),encoding='utf-8');print('DONE bytes',sum(p.stat().st_size for p in O.iterdir()),flush=True)
