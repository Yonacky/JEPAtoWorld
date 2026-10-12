"""Freeze original train/tune; sample previously unused sequence IDs before labels."""
from pathlib import Path
import argparse,json,zipfile,re,hashlib,numpy as np
from complex_causalworld_prepare import read_member,numeric_field,FIELDS,sha256_file
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--previous-prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prior-holdout',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
assert not (a.output/'prepared.npz').exists()
old=json.loads(a.previous.read_text());oldz=np.load(a.previous_prepared,allow_pickle=False)
assert sha256_file(a.archive)==old['archive_sha256']
assert sha256_file(a.previous_prepared)==old['output']['sha256']
prior=json.loads(a.prior_holdout.read_text())
used={v['episode'] for v in old['sequences']}|{0}|set(prior['new_evaluation'])
train=[v['episode'] for v in old['sequences'] if v['split']=='train']
tune=[v['episode'] for v in old['sequences'] if v['split']=='tune']
new=np.random.default_rng(20261015).permutation(sorted(set(range(200))-used))[:20].tolist()
frames=np.asarray(old['sampling']['frame_indices'],dtype=np.int32)
names=[f'{key}[{j}]' for key,width,_ in FIELDS for j in range(width)]
keep=[i for i,name in enumerate(names) if name in oldz['target_names'].tolist()]
assert [names[i] for i in keep]==oldz['target_names'].tolist()
manifest={'seed':20261015,'previous_manifest_sha256':sha256_file(a.previous),'prior_holdout_manifest_sha256':sha256_file(a.prior_holdout),'archive_sha256':old['archive_sha256'],'train':train,'tune':tune,'new_evaluation':new,'excluded_prior_ids':sorted(used),'frames':frames.tolist(),'camera':0,'members':{},'scope':'new sequence holdout, same data-generating environment; no physical intervention confirmation','protocol':'Selection fixed before reading new RGB/labels; all transform/readout hyperparameters use train/tune only'}
(a.output/'selection_before_read.json').write_text(json.dumps(manifest,indent=2))
items={k:[] for k in ['images','labels','split','episode','frame','episode_length']}
with zipfile.ZipFile(a.archive) as z:
 for split,ids in enumerate([train,tune,new]):
  for ep in ids:
   prefix=f'causal_world/train/seq_{ep:04d}/'
   with read_member(z,prefix+'observations.npz',manifest['members']) as obs:
    x=obs['obs'];assert x.shape==(1000,3,128,128,3) and x.dtype==np.uint8
    items['images'].append(x[frames,0].copy())
   with read_member(z,prefix+'step_infos.npz',manifest['members']) as info:
    y=np.concatenate([numeric_field(info,key,width)[frames] for key,width,_ in FIELDS],axis=1)[:,keep]
    items['labels'].append(y.astype(np.float32))
   items['split'].append(np.full(len(frames),split,dtype=np.int8));items['episode'].append(np.full(len(frames),ep,dtype=np.int32));items['frame'].append(frames);items['episode_length'].append(np.full(len(frames),1000,dtype=np.int32))
   print(split,ep,flush=True)
data={k:np.concatenate(v) for k,v in items.items()};data['target_names']=oldz['target_names'];data['split_names']=np.array(['train','tune','new_evaluation'])
for split in [0,1]:
 for k in ['images','labels','episode','frame','episode_length']:
  assert np.array_equal(data[k][data['split']==split],oldz[k][oldz['split']==split]),(split,k)
np.savez_compressed(a.output/'prepared.npz',**data)
manifest['prepared_sha256']=sha256_file(a.output/'prepared.npz');manifest['train_tune_exactly_match_previous']=True;manifest['split_counts']=[int(sum(data['split']==i)) for i in range(3)];manifest['source_sha256']=sha256_file(Path(__file__))
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('PREPARED',manifest['split_counts'],flush=True)
