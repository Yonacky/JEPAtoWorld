"""G7i sequential frozen generation and I-JEPA extraction; no model fitting."""
import argparse, pathlib, json, subprocess, sys, hashlib, os, time
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser()
 for n in ['output','study','model-root','sim-python','model-python']:p.add_argument('--'+n,type=pathlib.Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);s={'status':'running','started':time.time(),'completed':[],'device_isolation':os.environ.get('CUDA_VISIBLE_DEVICES')}
 def save(): (a.output/'status.json').write_text(json.dumps(s,indent=2))
 def run(stage,args,env=None):
  s['stage']=stage;save()
  with (a.output/(stage+'.log')).open('w') as log: subprocess.run(list(map(str,args)),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  s['completed'].append(stage);save()
 generator=pathlib.Path(__file__).with_name('generate_static_confirmation_20261013.py')
 extractor=a.study/'workspace/scripts/extract_confirmation_features.py'
 assert sha(extractor)=='24630778991c3aecb65bbc0a2740041c11d152781941f568a25c1cafac7364f7'
 sourcehash={str(f):sha(f) for f in [pathlib.Path(__file__),generator,extractor]};s['source_sha256']=sourcehash;save()
 try:
  for domain,seed,bg in [('id',2026101301,[.55,.95]),('ood',2026101302,[.25,.50])]:
   data=a.output/(domain+'_data');env=dict(os.environ);env['PYTHONPATH']=str(a.study/'workspace/CausalWorld-master');env['OMP_NUM_THREADS']='2'
   run(domain+'_generate',[a.sim_python,generator,'--legacy-script',a.study/'workspace/scripts/generate_tworef_id.py','--output',data,'--seed',seed,'--scenes',64,'--background',*bg],env)
   with np.load(data/'dataset.npz',allow_pickle=False) as f: z={k:f[k] for k in ['images','labels','scene','split','codes','target_names']}
   assert len(z['images'])==1728 and np.all(z['split']==2)
   frames=z['codes']@np.array([9,3,1]);np.testing.assert_array_equal(frames,np.tile(np.arange(27),64))
   prepared=a.output/(domain+'_prepared.npz');np.savez_compressed(prepared,images=z['images'],labels=z['labels'],split=z['split'],episode=z['scene'],frame=frames,episode_length=np.full(1728,27),target_names=z['target_names'])
   run(domain+'_extract',[a.model_python,extractor,'--root',a.study/'workspace','--model-root',a.model_root,'--prepared',prepared,'--output',a.output/(domain+'_features'),'--device','auto','--batch-size',16])
  assert sourcehash=={f:sha(f) for f in sourcehash}
  s.update(status='complete_pending_independent_data_audit',seconds=time.time()-s['started']);save()
 except BaseException as e:
  s.update(status='failed',error=repr(e));save();raise
if __name__=='__main__':main()
