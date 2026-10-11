import pathlib,json,hashlib
import numpy as np
import torch
TEST_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
from scipy import linalg as la
from sklearn.linear_model import Ridge
from e03_core import *

R=pathlib.Path(__file__).parent;rng=np.random.default_rng(731);results={}
def check(name,condition,details=None):
 assert condition,name
 results[name]={'passed':True,'details':details}
keys=list(SCHEMA['pinball']);a=np.zeros((3,len(keys)));a[:,keys.index('cyl_0_active')]=[.1,.5,.9];y,g=encode_targets('pinball',a,keys);gg=next(x for x in g if x['name']=='cyl_0_active');j=gg['slice'][0]
check('activation_continuous_preserved',gg['type']=='continuous' and np.allclose(y[:,j],[.1,.5,.9]))
a[0,keys.index('score')]=1.5
try:encode_targets('pinball',a,keys);caught=False
except AssertionError:caught=True
check('fractional_score_rejected',caught)
try:pair_indices(np.array([0,1,2,100,101]),1);caught=False
except AssertionError:caught=True
check('cross_segment_pair_rejected',caught)
p,f=pair_indices(np.arange(10,30),4);check('lag_direction',np.all(f-p==4))
x=rng.normal(size=(400,9));true=rng.normal(size=(9,3));y=x@true+3+rng.normal(size=(400,3))*.1;w,mx,my=ridge_fit(x,y,.03);ref=Ridge(alpha=.03*len(x),fit_intercept=True,solver='cholesky').fit(x,y)
err=float(np.max(np.abs((x-mx)@w+my-ref.predict(x))));check('ridge_matches_sklearn',err<1e-10,err)
q=la.qr(rng.normal(size=(9,9)))[0];wq,mq,yq=ridge_fit(x@q,y,.03);err=float(np.max(np.abs((x-mx)@w+my-(x@q-mq)@wq-yq)));check('ridge_rotation_invariance',err<1e-10,err)
center,pca,ev=pca_whiten(x,8);check('pca_whitening_identity',np.max(np.abs(((x-center)@pca).T@((x-center)@pca)/len(x)-np.eye(8)))<1e-8)
xx=[x[:200],x[200:300],x[300:]];yy=[y[:200],y[200:300],y[300:]];pg=[{'name':'synthetic','type':'continuous','slice':[0,3]}]
for budget in [4,'dense']:
 p,selected=probe(xx,yy,pg,budget);inds=selected['synthetic'];mu=yy[0].mean(0);sd=yy[0].std(0);best=np.inf
 for a in ALPHAS:
  ref=Ridge(alpha=a*len(xx[0]),fit_intercept=True,solver='cholesky').fit(xx[0][:,inds],(yy[0]-mu)/sd);loss=np.mean((ref.predict(xx[1][:,inds])-(yy[1]-mu)/sd)**2)
  if loss<best:best=loss;rp=ref.predict(xx[2][:,inds])*sd+mu
 check('optimized_probe_matches_sklearn_'+str(budget),np.max(np.abs(rp-p))<1e-10,float(np.max(np.abs(rp-p))))
angles=[.3,.6,1.,1.35];rho=.95;A=la.block_diag(*[rho*np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]) for t in angles]);mix=la.qr(rng.normal(size=(8,8)))[0];M=mix.T@A@mix
q,t,groups=schur_basis(M);check('schur_reconstruction',np.max(np.abs(M-q@t@q.T))<1e-10)
check('normal_cycle_schur_is_2d_blocks',all(len(g)==2 for g in groups) and np.linalg.norm(t*mask_for(groups,8))<1e-9)
check('symmetrization_cannot_distinguish_time_reverse',np.allclose((M+M.T)/2,(M.T+M)/2) and np.linalg.norm(M-M.T)>.5)
N=np.array([[.8,3.],[0,.5]]);qn,tn,gn=schur_basis(N);check('nonnormal_schur_not_block_diagonal',np.linalg.norm(tn*mask_for(gn,2))>1.)
C=np.stack([np.linalg.matrix_power(M,l) for l in [1,2,4,8]]);qo,info=optimize_joint(C,2,[np.eye(8)],steps=250,device=TEST_DEVICE);check('joint_optimizer_reduces_offblock_loss',info['final_objective']<info['initial_objective']*.5,info)
seq=np.zeros((13000,8));noise=np.sqrt(1-rho**2)
for i in range(1,len(seq)):seq[i]=seq[i-1]@M+rng.normal(size=8)*noise
seq=seq[1000:];train,tune,dev=seq[:8000],seq[8256:9256],seq[9512:11560];estimated=lag_cov(train,[1])[0]
check('empirical_cross_cov_orientation',np.linalg.norm(estimated-M)<np.linalg.norm(estimated-M.T)*.25,{'forward_error':float(np.linalg.norm(estimated-M)),'transpose_error':float(np.linalg.norm(estimated-M.T))})
Z=[a@q for a in [train,tune,dev]];X=[a[:-1] for a in Z];Y=[a[1:] for a in Z];errs={}
for name,gs in [('diagonal',groups_fixed(8,1)),('natural_blocks',groups),('dense',groups_fixed(8,8))]:
 p,meta=forecast_fit(X,Y,gs);errs[name]=float(np.mean((p-Y[2])**2))
check('known_cycles_need_blocks',errs['natural_blocks']<errs['diagonal']*.4 and abs(errs['natural_blocks']-errs['dense'])<.005,errs)
save={'test_device':TEST_DEVICE,'all_passed':True,'tests':results,'core_sha256':hashlib.sha256((R/'e03_core.py').read_bytes()).hexdigest(),'test_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'note':'Synthetic fixtures validate implementation, not evidence about JEPA.'}
(R/'selftest.json').write_text(json.dumps(save,indent=2));print('SELFTEST PASS',len(results),errs,flush=True)
