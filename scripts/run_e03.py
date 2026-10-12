"""Materialize immutable E03 run scripts, then execute explicit stages."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {"test": "e03_tests.py", "prepare": "e03_prepare.py", "fit": "e03_run.py", "validate": "validate_e03.py"}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["init", "features", *STAGES])
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--assets-root", type=Path)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    a = p.parse_args()
    run = a.run_dir.resolve()
    if a.stage == "init":
        run.mkdir(parents=True, exist_ok=False)
        for source in (ROOT / "core/e03").glob("*.py"):
            shutil.copy2(source, run / source.name)
        shutil.copy2(ROOT / "configs/e03.json", run / "protocol.json")
        print("Initialized", run)
        return
    if not (run / "protocol.json").is_file():
        p.error("run directory is not initialized")
    env = os.environ.copy()
    env["JEPA_DEVICE"] = a.device
    if a.stage == "prepare":
        if a.data_root is None or not a.data_root.is_dir():
            p.error("prepare requires an existing --data-root")
        env["JEPA_DATA_ROOT"] = str(a.data_root.resolve())
    if a.stage == "features":
        if a.assets_root is None:
            p.error("features requires --assets-root containing models and vendor")
        for name in ("models", "vendor"):
            source = (a.assets_root / name).resolve()
            if not source.is_dir():
                p.error("missing asset directory: " + str(source))
            target = run / name
            if target.exists() or target.is_symlink():
                if target.resolve() != source:
                    p.error("asset path mismatch: " + str(target))
            else:
                target.symlink_to(source, target_is_directory=True)
        for model in ("ijepa", "random", "dino"):
            subprocess.run([sys.executable, "-u", str(run / "features.py"), "--model", model, "--device", a.device], cwd=run, env=env, check=True)
    else:
        if a.stage == "validate" and not (run / "COMPLETE.json").is_file():
            p.error("fit has not completed; refusing to wait on an absent result")
        subprocess.run([sys.executable, "-u", str(run / STAGES[a.stage])], cwd=run, env=env, check=True)

if __name__ == "__main__":
    main()
