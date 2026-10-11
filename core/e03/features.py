import pathlib,sys,json,hashlib,time,types,argparse,shutil
import numpy as np
import torch
import torch.nn.functional as F
R=pathlib.Path(__file__).parent;sys.path.insert(0,str(R/'vendor/ijepa-main'))
from src.models.vision_transformer import vit_huge
from safetensors.torch import load_file
p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['ijepa','random','dino']);p.add_argument('--gpu',type=int,default=0);args=p.parse_args()
torch.set_num_threads(8);torch.manual_seed(0);torch.cuda.set_device(args.gpu);device=f'cuda:{args.gpu}'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def sdpa(self,x):
 b,n,c=x.shape;q,k,v=self.qkv(x).reshape(b,n,3,self.num_heads,c//self.num_heads).permute(2,0,3,1,4).unbind(0)
 y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,scale=self.scale)
 return self.proj_drop(self.proj(y.transpose(1,2).reshape(b,n,c))),None
meta={'model':args.model,'source_sha256':sha(pathlib.Path(__file__)),'protocol_sha256':sha(R/'protocol.json'),'data_manifest_sha256':sha(R/'data_manifest.json'),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(args.gpu),'seed':0,'resize':'torch bilinear antialias=True align_corners=False 224 square','channels':'RGB only','precision':'bfloat16 forward, float32 saved'}
if args.model in ['ijepa','random']:
 if args.model=='ijepa':
  start=time.time()
  while not (R/'models/ijepa_vith14_1k/VERIFIED.json').exists():
   if time.time()-start>3600:raise RuntimeError('checkpoint wait timeout')
   time.sleep(15)
 model=vit_huge(patch_size=14,img_size=[224]).eval()
 if args.model=='ijepa':
  ck=R/'models/ijepa_vith14_1k/model.safetensors';h=load_file(str(ck));d={};used=set()
  def put(dst,src):d[dst]=h[src];used.add(src)
  put('pos_embed','embeddings.position_embeddings')
  for suffix in ['weight','bias']:
   put('patch_embed.proj.'+suffix,'embeddings.patch_embeddings.projection.'+suffix);put('norm.'+suffix,'layernorm.'+suffix)
  for i in range(32):
   a=f'encoder.layer.{i}.';b=f'blocks.{i}.'
   for sf in ['weight','bias']:
    keys=[a+'attention.attention.'+x+'.'+sf for x in ['query','key','value']];d[b+'attn.qkv.'+sf]=torch.cat([h[k] for k in keys]);used.update(keys)
    for dst,src in [('attn.proj','attention.output.dense'),('norm1','layernorm_before'),('norm2','layernorm_after'),('mlp.fc1','intermediate.dense'),('mlp.fc2','output.dense')]:put(b+dst+'.'+sf,a+src+'.'+sf)
  assert used==set(h);model.load_state_dict(d,strict=True);meta['checkpoint']=json.loads((ck.parent/'VERIFIED.json').read_text());meta['state_dict_strict']=True;del h,d
 else:
  digest=hashlib.sha256()
  for k,v in model.state_dict().items():digest.update(k.encode());digest.update(v.detach().numpy().tobytes())
  meta['checkpoint']={'random_init_state_sha256':digest.hexdigest(),'seed':0,'architecture':'same as pretrained I-JEPA ViT-H/14'}
 att=model.blocks[0].attn;x=torch.randn(2,16,1280)
 with torch.no_grad():err=float((att(x)[0]-sdpa(att,x)[0]).abs().max())
 assert err<1e-5;meta['sdpa_float32_maxabs']=err
 model=model.to(device)
 full_input=torch.randn(1,3,224,224,device=device)
 with torch.inference_mode(): full_reference=model(full_input).float()
 for block in model.blocks:block.attn.forward=types.MethodType(sdpa,block.attn)
 with torch.inference_mode(): full_actual=model(full_input).float()
 full_error=float((full_actual-full_reference).abs().max())
 assert torch.allclose(full_actual,full_reference,atol=2e-4,rtol=2e-4),full_error
 meta['full_model_sdpa_check']={'maxabs':full_error,'atol':2e-4,'rtol':2e-4,'precision':'float32','input':'seeded synthetic image','reference':'original unpatched I-JEPA implementation'}
 del full_input,full_reference,full_actual
 mean=[.5]*3;std=[.5]*3;dim=1280
else:
 import timm
 from timm.models.vision_transformer import checkpoint_filter_fn
 model=timm.create_model('vit_small_patch14_dinov2',pretrained=False,dynamic_img_size=True).eval()
 ck=pathlib.Path('/home/kemove/dax-wm-assets/private/dinov2_vits14_pretrain.pth');state=torch.load(ck,map_location='cpu',weights_only=True)
 state=checkpoint_filter_fn(state,model);msg=model.load_state_dict(state,strict=False)
 assert not msg.missing_keys and not msg.unexpected_keys,str(msg)
 meta['checkpoint']={'path':str(ck),'sha256':sha(ck)};meta['state_dict_strict']=True;mean=[.485,.456,.406];std=[.229,.224,.225];dim=384
model=model.to(device).requires_grad_(False);mean=torch.tensor(mean,device=device)[None,:,None,None];std=torch.tensor(std,device=device)[None,:,None,None]
(R/'features').mkdir(exist_ok=True)
for dataset in ['pong','pinball']:
 stem=R/'features'/f'{dataset}_{args.model}'
 if stem.with_suffix('.json').exists():print('SKIP',stem,flush=True);continue
 z=np.load(R/'prepared'/f'{dataset}.npz',allow_pickle=False);imgs=z['images'];N=len(imgs);means=np.zeros((N,dim),np.float32);grids=np.zeros((N,dim*4),np.float32);start=time.time()
 with torch.inference_mode():
  for i in range(0,N,32):
   x=torch.from_numpy(imgs[i:i+32].copy()).permute(0,3,1,2).to(device=device,dtype=torch.float32)/255
   x=(F.interpolate(x,size=(224,224),mode='bilinear',align_corners=False,antialias=True)-mean)/std
   with torch.autocast('cuda',dtype=torch.bfloat16):
    y=model(x) if args.model!='dino' else model.forward_features(x)[:,1:]
   assert y.shape[1]==256
   y=y.float();means[i:i+len(y)]=y.mean(1).cpu().numpy()
   grid=y.reshape(len(y),16,16,dim).permute(0,3,1,2);grids[i:i+len(y)]=F.adaptive_avg_pool2d(grid,(2,2)).flatten(1).cpu().numpy()
   if i%1024==0:print('PROGRESS',dataset,args.model,i,N,round(time.time()-start,1),flush=True)
 assert np.isfinite(means).all() and np.isfinite(grids).all();assert np.std(means,axis=0).max()>1e-6
 if shutil.disk_usage(R).free<102*2**30:raise RuntimeError('SPACE_RESERVE')
 np.savez(stem.with_suffix('.npz'),mean=means,grid=grids)
 rec={**meta,'dataset':dataset,'n_frames':N,'dim':dim,'seconds':time.time()-start,'cache_sha256':sha(stem.with_suffix('.npz')),'peak_cuda_gib':torch.cuda.max_memory_allocated()/2**30}
 stem.with_suffix('.json').write_text(json.dumps(rec,indent=2));print('COMPLETE',dataset,args.model,rec['seconds'],flush=True)
