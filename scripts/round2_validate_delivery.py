from pathlib import Path
import hashlib, json, tarfile
r=Path(r'F:\2610\CVPR27')
p=r/'experiments/round2_first_batch.tar.gz'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='a677d604f197ae9ed0c0e0eb844dc04b819c81af1c9cee6a55f1b55ee0c5c128'
with tarfile.open(p) as t:
    for m in t.getmembers():
        target=(r/m.name).resolve()
        assert target.is_relative_to(r/'runs') and not m.issym() and not m.islnk(), m.name
        assert m.name.split('/')[1] in ['round2_a','round2_b','round2_evaluation','round2_execution']
    t.extractall(r, filter='data')
manifest=json.loads((r/'runs/round2_execution/manifest.json').read_text())
for name,expected in manifest['source_sha256'].items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==expected, name
for f in ['round2_a/COMPLETE_synthetic.json','round2_a/COMPLETE_real.json','round2_b/COMPLETE.json','round2_evaluation/COMPLETE.json']:
    print(f,json.loads((r/'runs'/f).read_text()))
(r/'runs/round2_execution/local_validation.json').write_text(json.dumps({'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_hashes_match':True,'completion_files_present':True,'safe_extraction':True},indent=2))
print('TRANSFER_AND_SOURCE_VALIDATED')
