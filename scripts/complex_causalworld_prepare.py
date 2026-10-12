#!/usr/bin/env python3
"""Prepare the fixed CausalWorld camera-0 pilot; no feature extraction.

Example: python complex_causalworld_prepare.py --archive causal_world.zip \
    --output pilot_causalworld --preview
Requires numpy; --preview additionally requires Pillow. ZIP members are read
in memory by exact names and never extracted. No intervention or privileged
optimized_images data are included in encoder inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

SEED = 20261011
SPLITS = (("train", 10), ("tune", 3), ("dev", 3))
N_FRAMES = 64
EPISODE_LENGTH = 1000
# Every field is independently shape-checked before flattening.
FIELDS = (
    ("tool_block/cartesian_position", 3, "position"),
    ("tool_block/color", 3, "color_diagnostic"),
    ("tool_block/linear_velocity", 3, "velocity_diagnostic_single_frame"),
    ("floor_color", 3, "color_diagnostic"),
    ("stage_color", 3, "color_diagnostic"),
    ("joint_positions", 9, "joint_position"),
)
AUDIT_FIELDS = ("tool_block/mass", "tool_block/friction", "floor_friction")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_member(archive: zipfile.ZipFile, name: str, records: dict):
    """Return an NPZ handle; reject ambiguous members and prohibit pickle."""
    data = archive.read(name)
    info = archive.getinfo(name)
    records[name] = {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data), "zip_crc32": f"{info.CRC:08x}",
    }
    return np.load(io.BytesIO(data), allow_pickle=False)


def numeric_field(npz, key: str, width: int | None = None) -> np.ndarray:
    value = npz[key]
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{key}: expected numeric data, got {value.dtype}")
    if width is None:
        if value.shape not in ((EPISODE_LENGTH,), (EPISODE_LENGTH, 1)):
            raise ValueError(f"{key}: expected scalar per frame; got {value.shape}")
        value = value.reshape(EPISODE_LENGTH, 1)
    elif value.shape != (EPISODE_LENGTH, width):
        raise ValueError(f"{key}: expected {(EPISODE_LENGTH, width)}; got {value.shape}")
    value = value.astype(np.float64)
    if not np.isfinite(value).all():
        raise ValueError(f"{key}: nonfinite target value")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory or explicit prepared.npz path")
    parser.add_argument("--preview", action="store_true",
                        help="Write first selected camera-0 RGB frame as PNG")
    args = parser.parse_args()
    output = args.output if args.output.suffix == ".npz" else args.output / "prepared.npz"
    manifest_path = output.with_suffix(".manifest.json") if args.output.suffix == ".npz" else output.parent / "manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Pilot outputs already exist; choose a new output path")
    if args.preview:
        from PIL import Image
    frames = np.linspace(0, EPISODE_LENGTH - 1, N_FRAMES, dtype=np.int32)
    assert len(np.unique(frames)) == N_FRAMES
    target_names = [f"{field}[{index}]" for field, width, _ in FIELDS for index in range(width)]
    roles = [role for _, width, role in FIELDS for _ in range(width)]
    manifest = {
        "schema_version": 1, "dataset": "BISCUIT CausalWorld",
        "archive_path": str(args.archive.resolve()),
        "archive_sha256": sha256_file(args.archive),
        "seed": SEED, "rng": "numpy.random.default_rng/PCG64",
        "encoder_input": "observations.npz:obs[:,0,:,:,:] uint8 RGB NHWC",
        "camera": 0, "source_episode_length": EPISODE_LENGTH,
        "sampling": {"method": "numpy.linspace(0,999,64,dtype=int32)",
                     "frames_per_episode": N_FRAMES, "frame_indices": frames.tolist()},
        "split_unit": "exclusive original source sequence",
        "split_mapping": {"train": 0, "tune": 1, "dev": 2},
        "members": {}, "sequences": [], "field_shapes": {},
        "notes": ["All splits sampled only from archive train sequences.",
                  "No official test evaluation; this is a fixed small pilot.",
                  "RGB color coordinates are correlated scalar diagnostics.",
                  "Linear velocity is diagnostic, not assumed recoverable from one frame.",
                  "No interventions, optimized_images, or other cameras enter images.",
                  "Mass/friction variance is audited, never inferred from intervention names.",
                  "Variance filtering uses sampled training targets only."],
    }
    images, labels, splits, episodes = [], [], [], []
    audit_values = {key: [] for key in AUDIT_FIELDS}
    audit_missing = {key: [] for key in AUDIT_FIELDS}
    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP member names are not permitted")
        sequence_ids = sorted(int(match.group(1)) for name in names
                              if (match := re.fullmatch(r"causal_world/train/seq_(\d{4})/observations\.npz", name)))
        if sequence_ids != list(range(200)):
            raise ValueError(f"Expected exactly train seq_0000..seq_0199; found {len(sequence_ids)}")
        selected = np.random.default_rng(SEED).permutation(sequence_ids)[:16]
        cursor = 0
        for split, count in SPLITS:
            for seq in selected[cursor:cursor + count]:
                seq = int(seq)
                prefix = f"causal_world/train/seq_{seq:04d}"
                with read_member(archive, f"{prefix}/observations.npz", manifest["members"]) as observation:
                    obs = observation["obs"]
                    if obs.shape != (EPISODE_LENGTH, 3, 128, 128, 3) or obs.dtype != np.uint8:
                        raise ValueError(f"{prefix}: unexpected obs shape/dtype {obs.shape}/{obs.dtype}")
                    images.append(obs[frames, 0].copy())
                    del obs
                with read_member(archive, f"{prefix}/step_infos.npz", manifest["members"]) as info:
                    missing = [key for key, _, _ in FIELDS if key not in info.files]
                    if missing:
                        raise KeyError(f"{prefix}: missing required fields {missing}; available={info.files}")
                    manifest["field_shapes"][f"seq_{seq:04d}"] = {
                        key: list(info[key].shape) for key, _, _ in FIELDS}
                    labels.append(np.concatenate([numeric_field(info, key, width)[frames]
                                                  for key, width, _ in FIELDS], axis=1))
                    if split == "train":
                        for key in AUDIT_FIELDS:
                            if key in info.files:
                                audit_values[key].append(numeric_field(info, key)[frames, 0])
                            else:
                                audit_missing[key].append(seq)
                splits.extend([split] * N_FRAMES)
                episodes.extend([seq] * N_FRAMES)
                manifest["sequences"].append({"sequence": f"seq_{seq:04d}", "episode": seq,
                                              "split": split, "frame_indices": frames.tolist()})
                print(f"Prepared {prefix} -> {split}", flush=True)
            cursor += count
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    splits = np.asarray(splits)
    train = labels[splits == "train"]
    variance = train.var(axis=0, dtype=np.float64)
    # Exact constancy avoids dropping physically small but genuinely varying factors.
    keep = np.ptp(train, axis=0) > 0
    manifest["targets"] = [
        {"name": name, "role": role, "train_variance": float(var),
         "train_min": float(train[:, i].min()), "train_max": float(train[:, i].max()),
         "retained": bool(keep[i]),
         "reason": "nonconstant_sampled_training_target" if keep[i] else "constant_in_sampled_training_targets"}
        for i, (name, role, var) in enumerate(zip(target_names, roles, variance))]
    manifest["physical_parameter_audit"] = {}
    for key in AUDIT_FIELDS:
        values = np.concatenate(audit_values[key]) if audit_values[key] else np.asarray([])
        complete = not audit_missing[key]
        manifest["physical_parameter_audit"][key] = {
            "train_variance": float(values.var()) if values.size else None,
            "train_min": float(values.min()) if values.size else None,
            "train_max": float(values.max()) if values.size else None,
            "missing_train_episodes": audit_missing[key],
            "complete": complete, "retained": False,
            "reason": "missing_training_field" if not complete else
                      "constant_in_sampled_training_targets" if np.ptp(values) == 0 else
                      "varying_but_audit_only_not_in_first_pilot_targets",
        }
    if not keep.any():
        raise ValueError("All candidate targets are constant in training")
    assert images.shape == (1024, 128, 128, 3)
    assert len(set(episodes)) == 16
    split_sets = {name: set(np.asarray(episodes)[splits == name].tolist()) for name, _ in SPLITS}
    assert all(not split_sets[a] & split_sets[b] for a in split_sets for b in split_sets if a != b)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, images=images, labels=labels[:, keep].astype(np.float32),
                        split=np.asarray([{"train": 0, "tune": 1, "dev": 2}[s] for s in splits], dtype=np.int8),
                        split_names=np.asarray(["train", "tune", "dev"]),
                        episode=np.asarray(episodes, dtype=np.int32),
                        frame=np.tile(frames, 16),
                        episode_length=np.full(1024, EPISODE_LENGTH, dtype=np.int32),
                        target_names=np.asarray(target_names)[keep])
    manifest["output"] = {"path": str(output.resolve()), "sha256": sha256_file(output),
                          "image_shape": list(images.shape), "label_shape": [1024, int(keep.sum())],
                          "split_counts": {name: int((splits == name).sum()) for name, _ in SPLITS},
                          "target_names": np.asarray(target_names)[keep].tolist()}
    if args.preview:
        preview = output.with_suffix(".firstframe.png")
        Image.fromarray(images[0], mode="RGB").save(preview)
        manifest["preview"] = {"path": str(preview), "episode": episodes[0], "frame": int(frames[0])}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest["output"], indent=2), flush=True)


if __name__ == "__main__":
    main()

