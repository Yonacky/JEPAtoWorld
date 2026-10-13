"""Extract frozen third-party LeJEPA image features with explicit provenance."""
import argparse, pathlib, json, hashlib, sys, time, os
import numpy as np
from family_image_adapter import FamilyImageAdapter

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint-dir',type=pathlib.Path,required=True);ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--device',default='auto');ap.add_argument('--batch-size',type=int,default=16);a=ap.parse_args();assert a.batch_size>0
 sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'core/e03'))
 from device_utils import activate_device
 import torch
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(0)
 dev=activate_device(a.device);a.output.mkdir(parents=True,exist_ok=False);start=time.time()
 adapter=FamilyImageAdapter(a.checkpoint_dir,str(dev))
 z=np.load(a.data,allow_pickle=False);ims=z['images'];assert ims.dtype==np.uint8 and ims.ndim==4 and ims.shape[-1]==3
 arrays={k:np.empty((len(ims),d),dtype=np.float32) for k,d in [('cls',768),('mean',768),('grid',3072)]};repeat=0.
 for i in range(0,len(ims),a.batch_size):
  x=adapter.preprocess(ims[i:i+a.batch_size]);y=adapter.encode(x)
  if i==0:
   yy=adapter.encode(x);repeat=max(float((y[k]-yy[k]).abs().max().cpu()) for k in arrays);assert repeat<=1e-6
  for k in arrays:
   v=y[k].cpu().numpy();assert np.isfinite(v).all();arrays[k][i:i+len(v)]=v
  if i%(a.batch_size*25)==0:print('encoded',min(i+a.batch_size,len(ims)),len(ims),flush=True)
 err=float(np.max(np.abs(arrays['grid'].reshape(-1,768,4).mean(2)-arrays['mean'])));assert err<1e-4
 np.savez_compressed(a.output/'features.npz',**arrays)
 md={k:z[k] for k in z.files if k not in ['images','actual_states']};np.savez_compressed(a.output/'sample_manifest.npz',**md)
 meta={**adapter.metadata,'source_dataset_sha256':sha(a.data),'source_script_sha256':sha(__file__),'adapter_sha256':sha(pathlib.Path(__file__).with_name('family_image_adapter.py')),'sample_count':len(ims),'features_sha256':sha(a.output/'features.npz'),'device':str(dev),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'device_uuid':str(getattr(torch.cuda.get_device_properties(dev),'uuid','unavailable')) if dev.type=='cuda' else None,'peak_cuda_bytes':torch.cuda.max_memory_allocated(dev) if dev.type=='cuda' else None,'repeat_maxabs':repeat,'mean_grid_maxabs':err,'precision':'FP32, TF32 disabled','seconds':time.time()-start,'comparison_scope':'Same raw RGB as I-JEPA, model-specific published preprocessing differs; not isolated pretraining-objective comparison'}
 (a.output/'config.json').write_text(json.dumps(meta,indent=2),encoding='utf8');(a.output/'COMPLETE.json').write_text(json.dumps({'status':'complete','sample_count':len(ims),'repeat_maxabs':repeat,'feature_shapes':{k:list(v.shape) for k,v in arrays.items()}},indent=2))
 (a.output/'manifest.json').write_text(json.dumps({p.name:sha(p) for p in a.output.iterdir() if p.is_file()},indent=2));print('COMPLETE',len(ims),flush=True)
if __name__=='__main__':main()