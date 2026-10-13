"""Run the frozen G7 video comparison serially after the data gate passes."""
import argparse, pathlib, json, subprocess, sys, time, hashlib, os

def main():
 p=argparse.ArgumentParser();p.add_argument('--family-root',type=pathlib.Path,required=True);p.add_argument('--study-root',type=pathlib.Path,required=True);p.add_argument('--model-root',type=pathlib.Path,required=True);p.add_argument('--device',default='auto');p.add_argument('--wait-seconds',type=int,default=7200);a=p.parse_args();r=a.family_root
 status=r/'video_pipeline_status.json';assert not status.exists(),'Preserve old execution; inspect before resuming'
 names=['verify_family_uniform_video_20261013.py','family_video_adapter.py','family_ijepa_video_baseline.py','complex_spatial_features.py','family_ratio_readout.py','device_utils.py']
 hashes={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names};state={'status':'waiting_for_uniform_data','started_unix':time.time(),'source_sha256':hashes,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'completed_stages':[]}
 def save():
  q=status.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2));q.replace(status)
 def run(stage,script,*args):
  state.update(status='running',stage=stage);save();print('STAGE',stage,flush=True)
  subprocess.run([sys.executable,str(r/script),*map(str,args)],check=True)
  state['completed_stages'].append(stage);save()
 save();data=r/'uniform_production40'
 try:
  while not (data/'validation.json').exists():
   if (data/'FAILED.json').exists():raise RuntimeError('Uniform generation failed; no encoder execution')
   if time.time()-state['started_unix']>a.wait_seconds:raise TimeoutError('Uniform data not ready within bound')
   time.sleep(10)
  assert json.loads((data/'validation.json').read_text())['status']=='PASS_INVARIANTS'
  v=r/'vjepa2_production40'
  state.update(status='waiting_for_external_vjepa2',stage='external_audit_and_encoding');save()
  while not (v/'COMPLETE.json').exists():
   if time.time()-state['started_unix']>a.wait_seconds:raise TimeoutError('External audited extraction not complete within bound')
   time.sleep(10)
  audit=json.loads((data/'resampling_audit_v2.json').read_text())
  assert audit['status']=='passed' and audit['scenes']==40
  assert audit['sha256'][str(data/'dataset.npz')]==hashlib.sha256((data/'dataset.npz').read_bytes()).hexdigest()
  state['completed_stages'] += ['external_uniform_full_source_audit','external_vjepa2_encode'];save()
  assert json.loads((v/'COMPLETE.json').read_text())['episode_count']==720
  run('vjepa2_fixed_readout','family_ratio_readout.py','--data',data/'dataset.npz','--features',v/'features.npz','--dataset-manifest',v/'sample_manifest.npz','--window-spec',v/'window_spec.json','--input-kind','branch_vector','--output',r/'vjepa2_ratio_readout')
  b=r/'ijepa_same_video40'
  run('ijepa_same_observation_encode','family_ijepa_video_baseline.py','--dataset',data/'dataset.npz','--model-root',a.model_root,'--root',r,'--output',b,'--device',a.device,'--batch-size',16)
  run('ijepa_same_observation_readout','family_ratio_readout.py','--data',data/'dataset.npz','--features',b/'features.npz','--dataset-manifest',b/'sample_manifest.npz','--window-spec',b/'window_spec.json','--input-kind','branch_vector','--output',r/'ijepa_same_video_ratio_readout')
  assert hashes=={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in names},'Source changed during run'
  state.update(status='complete',seconds=time.time()-state['started_unix']);save();print('PIPELINE_COMPLETE',flush=True)
 except Exception as e:
  state.update(status='failed',error=repr(e));save();raise
if __name__=='__main__':main()