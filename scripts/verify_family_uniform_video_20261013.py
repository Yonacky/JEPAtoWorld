"""Check uniform RGB resampling leaves historical G5f physics unchanged."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--reference',type=Path,required=True);ap.add_argument('--uniform',type=Path,required=True);a=ap.parse_args()
    oldcfg=json.loads((a.reference/'config.json').read_text());cfg=json.loads((a.uniform/'config.json').read_text())
    for k in ['seed','scenes','mass_levels','force_levels','friction','modes','steps','force_steps','scene_split']: assert cfg[k]==oldcfg[k],k
    steps=cfg['record_steps'];assert steps==list(range(0,121,8))
    old=np.load(a.reference/'full_trajectories_with_repeats.npz');new=np.load(a.uniform/'full_trajectories_with_repeats.npz');assert set(old.files)==set(new.files)
    checks=[]
    for key in old.files:
        if key.endswith('_images'):
            shared=sorted(set(oldcfg['record_steps'])&set(steps))
            for t in shared:np.testing.assert_array_equal(old[key][oldcfg['record_steps'].index(t)],new[key][steps.index(t)])
        else:np.testing.assert_array_equal(old[key],new[key])
        checks.append(key)
    records=json.loads((a.uniform/'records.json').read_text());lookup={(int(v['scene']),v['contact_mode'],v['mass_kg'],v['force_n']):v['key'] for v in records if v['repeat']==0}
    equality=[]
    for scene in range(cfg['scenes']):
        for mode in ['on','off']:
            for ma,fa,mb,fb in [(.01,.01,.02,.02),(.02,.01,.04,.02)]:
                ka,kb=lookup[(scene,mode,ma,fa)],lookup[(scene,mode,mb,fb)]
                equal=bool(np.array_equal(new[ka+'_images'],new[kb+'_images']))
                if mode=='off':assert equal
                equality.append({'scene':scene,'mode':mode,'keys':[ka,kb],'all16_images_equal':equal})
    z=np.load(a.uniform/'dataset.npz');assert len(z['images'])==cfg['scenes']*18*16
    old_records=json.loads((a.reference/'records.json').read_text())
    old_map={v['key']:v for v in old_records}
    for rec in records:
        assert rec['actual_dynamics']==old_map[rec['key']]['actual_dynamics']
        if rec['repeat']!=0:continue
        ids=np.flatnonzero(z['episode']==rec['key']);ids=ids[np.argsort(z['frame'][ids])];assert len(ids)==16
        np.testing.assert_array_equal(z['images'][ids],new[rec['key']+'_images'])
        np.testing.assert_array_equal(z['actual_states'][ids],new[rec['key']+'_fullstates'][steps])
        scene=rec['scene'];mi=cfg['mass_levels'].index(rec['mass_kg']);fi=cfg['force_levels'].index(rec['force_n'])
        for k,v in {'scene':scene,'contact_mode':rec['contact_mode'],'mass_level':mi,'force_level':fi,'split':0 if scene<24 else 1 if scene<32 else 2,'episode_length':121}.items():assert np.all(z[k][ids]==v),k
        np.testing.assert_array_equal(z['labels'][ids],np.tile([rec['mass_kg'],cfg['friction'],rec['force_n']],(16,1)))
    for ep in np.unique(z['episode']):np.testing.assert_array_equal(z['frame'][z['episode']==ep],steps)
    assert np.allclose(z['time_seconds'],z['frame']*.004,atol=0,rtol=0)
    files=[a.reference/'full_trajectories_with_repeats.npz',a.reference/'config.json',a.uniform/'full_trajectories_with_repeats.npz',a.uniform/'config.json',a.uniform/'dataset.npz',Path(__file__)]
    result={'status':'passed','scenes':cfg['scenes'],'arrays_compared':len(checks),'saved_tool_and_robot_states_exact':True,'dataset_row_source_mapping_exact':True,'actual_dynamics_exact':True,'full_contact_records_exact':records==old_records,'shared_rgb_steps':shared,'shared_rgb_exact':True,'uniform_steps':steps,'seconds_per_frame':.032,'paired_image_checks':equality,'sha256':{str(p):sha(p) for p in files},'scope':'Rendering resampling audit; not new independent scientific scenes or model result'}
    out=a.uniform/'resampling_audit_v2.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['sha256','paired_image_checks']}),flush=True)
if __name__=='__main__':main()