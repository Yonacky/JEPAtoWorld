"""Extract frozen I-JEPA mean and spatial 2x2 features from unchanged prepared RGB."""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import types

import numpy as np

SEED = 20261011
LAYOUT = "channel-major: N,1280,2,2 flattened to N,5120; cells TL,TR,BL,BR"
REQUIRED = ("images", "labels", "split", "episode", "frame", "episode_length", "target_names")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(b)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def validate(data):
    assert set(REQUIRED).issubset(data), "Missing prepared fields"
    images, labels = data["images"], data["labels"]
    n = len(images)
    assert n > 0 and images.ndim == 4 and images.shape[-1] == 3
    assert images.dtype == np.uint8 and min(images.shape[1:3]) > 0
    assert labels.ndim == 2 and len(labels) == n and np.isfinite(labels).all()
    assert len(data["target_names"]) == labels.shape[1]
    assert set(np.unique(data["split"])) in ({0, 1, 2}, {2}), "Allow test-only frozen extraction"
    for k in ("split", "episode", "frame", "episode_length"):
        assert data[k].shape == (n,), k
    groups = [set(data["episode"][data["split"] == s].tolist()) for s in range(3)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for ep in np.unique(data["episode"]):
        ids = np.flatnonzero(data["episode"] == ep)
        assert np.all(np.diff(data["frame"][ids]) > 0)
        assert len(np.unique(data["episode_length"][ids])) == 1
    assert np.all(data["frame"] >= 0)
    assert np.all(data["frame"] < data["episode_length"])


def device_metadata(torch, device):
    meta = {
        "selected_device": str(device),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
        "visible_cuda_count": torch.cuda.device_count(),
        "selected_device_uuid": None,
    }
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(device)
        meta.update(selected_device_name=prop.name,
                    selected_device_uuid=str(getattr(prop, "uuid", "unavailable")),
                    selected_device_total_memory=int(prop.total_memory),
                    selected_logical_index=device.index)
        assert meta["selected_device_uuid"] != "unavailable", "CUDA UUID unavailable"
    return meta


def provenance(args, data):
    vendor = args.model_root / "vendor/ijepa-main"
    vendor_files = sorted((vendor / "src").rglob("*.py"))
    assert vendor_files, "Missing I-JEPA vendor source"
    vendor_hashes = {p.relative_to(vendor).as_posix(): sha(p) for p in vendor_files}
    return {
        "prepared_sha256": sha(args.prepared),
        "images_sha256": array_sha(data["images"]),
        "images_shape": list(data["images"].shape),
        "images_dtype": str(data["images"].dtype),
        "checkpoint_sha256": sha(args.model_root / "models/ijepa_vith14_1k/model.safetensors"),
        "source_script_sha256": sha(pathlib.Path(__file__)),
        "device_utils_sha256": sha(args.root / "core/e03/device_utils.py"),
        "vendor_source_sha256": hashlib.sha256(json.dumps(vendor_hashes, sort_keys=True).encode()).hexdigest(),
        "vendor_source_files": vendor_hashes,
        "manifest_hashes": {k: {"sha256": array_sha(data[k]), "shape": list(data[k].shape),
                               "dtype": str(data[k].dtype)} for k in REQUIRED if k != "images"},
        "image_orientation": "Prepared camera RGB bytes consumed unchanged; no flip or rotation",
        "precision": "FP32, TF32 disabled",
        "resize": "224 square bilinear antialias=True align_corners=False",
        "normalization": "RGB (x/255 - .5)/.5",
        "grid_layout": LAYOUT,
        "seed": SEED,
    }


def check_features(mean, grid, n):
    assert mean.shape == (n, 1280) and grid.shape == (n, 5120)
    assert mean.dtype == grid.dtype == np.float32
    assert np.isfinite(mean).all() and np.isfinite(grid).all()
    reconstructed = grid.reshape(n, 1280, 4).mean(-1)
    err = float(np.max(np.abs(mean - reconstructed)))
    assert np.allclose(mean, reconstructed, atol=2e-5, rtol=2e-5), err
    return err


def extract(images, model_root, device, batch):
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file
    sys.path.insert(0, str(model_root / "vendor/ijepa-main"))
    from src.models.vision_transformer import vit_huge

    torch.manual_seed(SEED)
    ck = model_root / "models/ijepa_vith14_1k/model.safetensors"
    h = load_file(str(ck))
    d, used = {}, set()

    def put(dst, src):
        d[dst] = h[src]
        used.add(src)

    put("pos_embed", "embeddings.position_embeddings")
    for sf in ("weight", "bias"):
        put("patch_embed.proj." + sf, "embeddings.patch_embeddings.projection." + sf)
        put("norm." + sf, "layernorm." + sf)
    for i in range(32):
        a, b = f"encoder.layer.{i}.", f"blocks.{i}."
        for sf in ("weight", "bias"):
            keys = [a + "attention.attention." + x + "." + sf for x in ("query", "key", "value")]
            d[b + "attn.qkv." + sf] = torch.cat([h[k] for k in keys])
            used.update(keys)
            for dst, src in (("attn.proj", "attention.output.dense"),
                             ("norm1", "layernorm_before"), ("norm2", "layernorm_after"),
                             ("mlp.fc1", "intermediate.dense"), ("mlp.fc2", "output.dense")):
                put(b + dst + "." + sf, a + src + "." + sf)
    assert used == set(h), "Unused checkpoint keys"
    model = vit_huge(patch_size=14, img_size=[224]).float().eval()
    model.load_state_dict(d, strict=True)
    del h, d
    model.requires_grad_(False)

    def sdpa(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0., scale=self.scale)
        return self.proj_drop(self.proj(y.transpose(1, 2).reshape(b, n, c))), None

    att = model.blocks[0].attn
    x = torch.randn(2, 16, 1280)
    with torch.no_grad():
        err = float((att(x)[0] - sdpa(att, x)[0]).abs().max())
    assert err < 1e-5, err
    model = model.to(device)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.inference_mode():
        reference = model(x)
    for block in model.blocks:
        block.attn.forward = types.MethodType(sdpa, block.attn)
    with torch.inference_mode():
        actual = model(x)
    fullerr = float((reference - actual).abs().max())
    assert torch.allclose(actual, reference, atol=2e-4, rtol=2e-4), fullerr
    del reference, actual, x
    mean = np.empty((len(images), 1280), np.float32)
    grid = np.empty((len(images), 5120), np.float32)
    with torch.inference_mode():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i:i + batch].copy()).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255
            x = (F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True) - .5) / .5
            y = model(x)
            assert y.shape[1:] == (256, 1280) and y.dtype == torch.float32
            spatial = y.reshape(len(y), 16, 16, 1280).permute(0, 3, 1, 2)
            pooled = F.adaptive_avg_pool2d(spatial, (2, 2))
            mean[i:i + len(y)] = y.mean(1).cpu().numpy()
            grid[i:i + len(y)] = pooled.flatten(1).cpu().numpy()
            print("features", i + len(y), len(images), flush=True)
    consistency = check_features(mean, grid, len(images))
    meta = {
        "strict_load": True, "all_checkpoint_keys_used": True,
        "sdpa_attention_maxabs": err, "sdpa_full_model_maxabs": fullerr,
        "sdpa_full_atol_rtol": 2e-4, "mean_grid_maxabs": consistency,
        "mean_grid_atol_rtol": 2e-5,
        "pooling": "mean over 256 tokens and adaptive average pooling 16x16 token map to 2x2",
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mean, grid, meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "model-root", "prepared", "output"):
        p.add_argument("--" + name, type=pathlib.Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--features-cache", type=pathlib.Path,
                   help="Explicit cache only; requires matching metadata, archive hash, and sample manifest")
    args = p.parse_args()
    assert args.batch_size > 0
    start = time.time()
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    sys.path.insert(0, str(args.root / "core/e03"))
    from device_utils import activate_device
    device = activate_device(args.device)
    with np.load(args.prepared, allow_pickle=False) as z:
        data = {k: z[k] for k in REQUIRED}
    validate(data)
    expected = provenance(args, data)
    config = {
        **expected, **device_metadata(torch, device),
        "torch": torch.__version__, "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "args": {k: str(v) if isinstance(v, pathlib.Path) else v for k, v in vars(args).items()},
        "split_frame_counts": [int(np.sum(data["split"] == s)) for s in range(3)],
        "split_episode_counts": [len(np.unique(data["episode"][data["split"] == s])) for s in range(3)],
    }
    if args.features_cache:
        cache = args.features_cache
        meta = json.loads((cache.parent / "features.json").read_text(encoding="utf-8"))
        for key, value in expected.items():
            assert meta.get(key) == value, ("Feature cache provenance mismatch", key)
        assert meta["strict_load"] and meta["all_checkpoint_keys_used"]
        assert meta["features_sha256"] == sha(cache), "Feature cache archive hash mismatch"
        with np.load(cache.parent / "sample_manifest.npz", allow_pickle=False) as old:
            assert set(old.files) == set(REQUIRED) - {"images"}
            for k in old.files:
                assert old[k].dtype == data[k].dtype and np.array_equal(old[k], data[k]), k
        with np.load(cache, allow_pickle=False) as z:
            mean, grid = z["mean"], z["grid"]
        error = check_features(mean, grid, len(data["images"]))
        assert error == meta["mean_grid_maxabs"]
        config["reused_features_sha256"] = sha(cache)
        config["feature_extraction_device"] = meta.get("extraction_device")
    else:
        mean, grid, checks = extract(data["images"], args.model_root, device, args.batch_size)
        meta = {**expected, **checks, "extraction_device": device_metadata(torch, device),
                "torch": torch.__version__, "numpy": np.__version__, "cuda_runtime": torch.version.cuda}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / "sample_manifest.npz", **{k: data[k] for k in REQUIRED if k != "images"})
    np.savez(args.output / "features.npz", mean=mean, grid=grid)
    meta["features_sha256"] = sha(args.output / "features.npz")
    dump(args.output / "features.json", meta)
    dump(args.output / "config.json", config)
    dump(args.output / "COMPLETE.json", {
        "status": "frozen_spatial_feature_extraction_complete", "seconds": time.time() - start,
        "source_script_sha256": expected["source_script_sha256"],
        "features_sha256": meta["features_sha256"], "mean_grid_maxabs": meta["mean_grid_maxabs"],
        "sample_count": len(mean), "mean_shape": list(mean.shape), "grid_shape": list(grid.shape),
    })


if __name__ == "__main__":
    main()

