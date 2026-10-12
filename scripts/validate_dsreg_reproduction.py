"""Audit completed official DSReg runs without importing third-party training code."""
import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[4, 8])
    args = parser.parse_args()
    source_file = args.source / "experiments/synthetic/run.py"
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    protocol = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROTOCOL" for t in node.targets))
    with (args.source / "experiments/synthetic/paper_results.csv").open(encoding="utf-8", newline="") as handle:
        reference = {int(row["N"]): row for row in csv.DictReader(handle)}
    checks = []
    for dimension in args.dimensions:
        path = args.results / f"N{dimension}_seed0.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["N"] == dimension and result["seed"] == 0
        differences = {k: [v, result.get(k)] for k, v in protocol.items() if result.get(k) != v}
        assert not differences, differences
        assert len(result["restart_final_losses"]) == protocol["encoder_restarts"]
        assert all(math.isfinite(result[k]) for k in ("lejepa_mcc", "dsreg_mcc", "lejepa_r2", "dsreg_r2"))
        error = abs(result["lejepa_r2"] - result["dsreg_r2"])
        assert error < 1e-5, error
        row = reference[dimension]
        checks.append({"dimension": dimension, "seed": 0, "official_defaults_match": True,
                       "dense_r2_invariance_error": error, "mcc_before": result["lejepa_mcc"],
                       "mcc_after": result["dsreg_mcc"], "dense_r2": result["dsreg_r2"],
                       "paper_mcc_mean": float(row["posthoc_mcc_mean"]),
                       "paper_mcc_std": float(row["posthoc_mcc_std"]), "paper_seed_count": int(row["n"]),
                       "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    output = {"scope": "Official protocol subset; not bitwise or full seed-sweep reproduction",
              "source_run_sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(), "checks": checks,
              "dense_metric_caveat": "Official dense_r2 fits and scores on held-out states; a span diagnostic, not cross-split readout generalization"}
    (args.results / "local_validation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))

if __name__ == "__main__":
    main()
