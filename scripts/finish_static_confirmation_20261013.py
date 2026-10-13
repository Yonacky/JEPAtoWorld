"""G7i data gate, frozen evaluation, independent audit, sequential CPU only."""
import argparse,pathlib,json,subprocess,sys,time,hashlib
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--study',type=pathlib.Path,required=True);a=p.parse_args();scripts=pathlib.Path(__file__).parent
status=a.root/'downstream_status.json';assert not status.exists();s={'status':'waiting_for_generation','started':time.time(),'completed':[]}
def save():status.write_text(json.dumps(s,indent=2))
def run(name,script,*args):
 s.update(status='running',stage=name);save()
 with (a.root/(name+'.log')).open('w') as f:subprocess.run([sys.executable,str(scripts/script),*map(str,args)],stdout=f,stderr=subprocess.STDOUT,check=True)
 s['completed'].append(name);save()
save()
try:
 while True:
  upstream=json.loads((a.root/'status.json').read_text())
  if upstream['status']=='failed':raise RuntimeError('Upstream failed')
  if upstream['status']=='complete_pending_independent_data_audit':break
  if time.time()-s['started']>2700:raise TimeoutError('Generation bound exceeded')
  time.sleep(10)
 old=[a.study/x/'contexts.json' for x in ['pilot','confirmation','step3b_data','repair_data','oat_data','tworef_id_data','tworef_ood_data']]
 for domain,seed,bg,peer in [('id',2026101301,[.55,.95],'ood'),('ood',2026101302,[.25,.50],'id')]:
  run(domain+'_data_audit','verify_static_confirmation_20261013.py','--data',a.root/(domain+'_data'),'--output',a.root/(domain+'_independent_audit'),'--seed',seed,'--scenes',64,'--background',*bg,'--old-contexts',*old,'--peer-contexts',a.root/(peer+'_data/contexts.json'))
 for domain in ['id','ood']:
  out=a.root/(domain+'_evaluation')
  run(domain+'_evaluate','evaluate_frozen_oblique_20261013.py','--data',a.root/(domain+'_data/dataset.npz'),'--features',a.root/(domain+'_features/features.npz'),'--models',scripts/'models.npz','--joint0',a.study/'step2b/fitted.npz','--selection',scripts/'selection.json','--frozen-config',scripts/'frozen_config.json','--output',out,'--domain',domain)
  run(domain+'_evaluation_audit','audit_frozen_oblique_confirmation_20261013.py','--evaluation',out)
 s.update(status='complete',seconds=time.time()-s['started']);save()
except BaseException as e:
 s.update(status='failed',error=repr(e));save();raise
