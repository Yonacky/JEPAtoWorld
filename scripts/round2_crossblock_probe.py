"""Budget-matched supervised coordinate diagnostics on frozen unsupervised transforms."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy import linalg as la
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from round2_b_shared_private import ridge, r2, write, paired_ci

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--device',default='auto');ap.add_argument('--bootstrap',type=int,default=200);args=ap.parse_args()
    O=args.output;O.mkdir(parents=True,exist_ok=True)
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(2)
    write(O/'config.json',{'scope':'Supervised coordinate-selection diagnostic only; no new unsupervised representation fitting; reused r0 exploration','root':str(args.root),'device':device,'budgets':[1,2,4,8],'methods':['white','sfa','full_block4'],'selection':'Per-target train standardized cross-covariance squared; select global top-k or complete single four-coordinate block. Tune selects ridge independently; no dev selection.','capacity':'Only top4 versus complete block4 is strictly matched input width. Different methods and budgets are descriptive, not dev-selected winners.','bootstrap':'Paired contiguous blocks of 64 dev frames; conditional on reused r0','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'ridge_code_sha256':hashlib.sha256(Path(__file__).with_name('round2_b_shared_private.py').read_bytes()).hexdigest()})
    results=[];selection=[];comparisons=[];audits=[]
    for ds,ids,names in [('pong',[4,5],['paddle_left_y','paddle_right_y']),('pinball',[4,6,7,8],['cyl_0_active','cyl_2_active','cyl_3_active','cyl_4_active'])]:
        I=args.root/'runs/e04_inputs';cp=I/'cache'/f'{ds}_ijepa_r0.npz';z=np.load(cp)
        schema_path=args.root/f'experiments/e03_fp32_20261011/{ds}_factor_schema.json';schema=json.loads(schema_path.read_text());idx={g['name']:g['slice'][0] for g in schema if g['type']=='continuous'};assert [idx[n] for n in names]==ids
        X=[z[s] for s in ['train','tune','dev']];S=[z['source_'+s] for s in ['train','tune','dev']]
        assert all(np.all(np.diff(s)>0) for s in S)
        assert not any(np.intersect1d(S[i],S[j]).size for i in range(3) for j in range(i))
        raw=[z['target_'+s][:,ids] for s in ['train','tune','dev']];mean,std=raw[0].mean(0),raw[0].std(0);assert (std>1e-8).all();Y0=[(y-mean)/std for y in raw];H=la.helmert(len(ids),full=False).T
        Y=[np.column_stack([y,y.mean(1),y@H]) for y in Y0];targets=names+['common_standardized_mean']+[f'helmert_difference_{j}' for j in range(H.shape[1])]
        audits.append({'dataset':ds,'cache_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'schema_sha256':hashlib.sha256(schema_path.read_bytes()).hexdigest(),'source_hashes':[hashlib.sha256(s.tobytes()).hexdigest() for s in S],'target_names':targets,'factor_train_mean':mean.tolist(),'factor_train_std':std.tolist(),'helmert':H.tolist()})
        for method in ['white','sfa','full_block4']:
            tp=I/'transforms'/f'{ds}_ijepa_r0_{method}.npz';q=np.load(tp)['past_basis'];assert q.shape==(128,128)
            orth_error=float(np.max(np.abs(q.T@q-np.eye(128))));assert orth_error<1e-6
            A=[x@q for x in X];xc=A[0]-A[0].mean(0);ym,ys=Y[0].mean(0),Y[0].std(0);assert (ys>1e-8).all();cross=xc.T@((Y[0]-ym)/ys)/len(xc);scores=cross**2
            pred={name:np.zeros_like(Y[2]) for name in ['top1','top2','top4','top8','single_block4']}
            chosen={name:[] for name in pred};alphas={name:[] for name in pred}
            for j,target in enumerate(targets):
                # Stable index tie-break ensures deterministic selection on equal scores.
                order=np.argsort(-scores[:,j],kind='stable');bs=scores[:,j].reshape(32,4).sum(1);block=int(bs.argmax())
                for name in pred:
                    coords=np.arange(block*4,block*4+4) if name=='single_block4' else order[:int(name[3:])]
                    inputs=[a[:,coords] for a in A];yy=[y[:,j:j+1] for y in Y];p,alpha,loss=ridge(inputs,yy,device);pred[name][:,j]=p[:,0];chosen[name].append(coords.tolist());alphas[name].append(float(alpha[0]))
                    gram=np.cov(inputs[0],rowvar=False,bias=True);ge=np.linalg.eigvalsh(np.atleast_2d(gram));edf=float(np.sum(np.maximum(ge,0)/(np.maximum(ge,0)+alpha[0])))
                    selection.append({'dataset':ds,'method':method,'diagnostic':name,'target':target,'indices':coords.tolist(),'parent_blocks':(coords//4).tolist(),'unique_blocks':int(len(np.unique(coords//4))),'train_score_sum':float(scores[coords,j].sum()),'alpha':float(alpha[0]),'tune_mse':float(loss[0]),'ridge_effective_df_excluding_intercept':edf})
            for name,p in pred.items():
                np.savez_compressed(O/f'{ds}_{method}_{name}.npz',prediction=p,truth=Y[2],source_dev=S[2],target_names=np.asarray(targets),indices=np.asarray(chosen[name]),alpha=np.asarray(alphas[name]),transform_sha256=np.asarray(hashlib.sha256(tp.read_bytes()).hexdigest()))
                for j,target in enumerate(targets):results.append({'dataset':ds,'method':method,'diagnostic':name,'target':target,'dev_r2':float(r2(Y[2],p)[j]),'coordinates':chosen[name][j],'alpha':alphas[name][j]})
            comparisons.append({'dataset':ds,'method':method,'method_diagnostic':'top4','reference':'single_block4','targets':targets,'delta_r2':(r2(Y[2],pred['top4'])-r2(Y[2],pred['single_block4'])).tolist(),'paired_block_ci95':paired_ci(Y[2],pred['top4'],pred['single_block4'],args.bootstrap,777)})
            print(ds,method,'complete',flush=True)
        write(O/'results.json',results);write(O/'selection.json',selection);write(O/'comparisons.json',comparisons);write(O/'audits.json',audits)
    report=['全局少量坐标信息定位：坐标/整块均由训练标签选择，属于监督诊断，不是新无监督方法。','white/SFA/full_block4为冻结输入变换；top1/2/4/8按每目标训练关联能量选坐标，single_block4按同能量选完整块。仅top4与single_block4直接匹配4维。','对full_block4，unique_blocks可用于判断信息是否跨既有块；white/SFA的每4维只是固定连续分组，没有已有机制块含义。','r0已反复使用；配对时间块区间不是独立确认，坐标选择对dev不知情也不会恢复确认资格。']
    for row in results:
        if 'difference' in row['target'] or 'common' in row['target']:report.append(f"{row['dataset']} {row['method']} {row['diagnostic']} {row['target']}: R2={row['dev_r2']:.5f}, coords={row['coordinates']}")
    (O/'report.txt').write_text('\n'.join(report),encoding='utf-8');write(O/'COMPLETE.json',{'status':'complete','rows':len(results),'selection_rows':len(selection),'scope':'supervised exploratory diagnostic only'})
if __name__=='__main__':main()
