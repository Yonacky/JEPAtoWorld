"""Audited OK-AI LeJEPA reproduction backbone, not original-author weights.

Requires torch, torchvision, timm==1.0.19, safetensors and Pillow.
No remote custom code is executed. Explicitly omits six training-projector tensors.
"""
import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = 'OK-AI/lejepa-vitb16-pretrain-in1k'
REVISION = '5dd71e2fc9a8e97d84c047f3a8cd5bf19ff199fa'
WEIGHT_SHA256 = '5dd54eaa80e3160420a9b1682b2cd2e9505aff6566d1f54b742067fef149f85a'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class FamilyImageAdapter:
    def __init__(self, checkpoint_dir, device='cpu'):
        import torch
        import timm
        from safetensors.torch import load_file
        from torchvision import transforms as T
        from torchvision.transforms import InterpolationMode
        self.torch = torch
        p = Path(checkpoint_dir)
        cfg = json.loads((p / 'config.json').read_text())
        expected = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                        mlp_ratio=4, num_register_tokens=0, init_values=None)
        for k, v in expected.items():
            if cfg.get(k) != v:
                raise ValueError((k, cfg.get(k), v))
        if cfg.get('image_size', cfg.get('img_size', 224)) != 224:
            raise ValueError('Only fixed 224 checkpoint supported')
        weight = p / 'model.safetensors'
        digest = sha256(weight)
        if digest != WEIGHT_SHA256:
            raise ValueError('Checkpoint SHA256 differs from frozen revision')
        sd = load_file(str(weight), device='cpu')
        if not all(k.startswith('backbone.') for k in sd):
            raise ValueError('Unexpected key prefix')
        sd = {k[len('backbone.'):]: v for k, v in sd.items()}
        projector = {}
        for idx, outdim, indim in [(0, 2048, 768), (2, 2048, 2048), (4, 64, 2048)]:
            for suffix, shape in [('weight', (outdim, indim)), ('bias', (outdim,))]:
                k = f'cva_module_proj.mlp.{idx}.{suffix}'
                if k not in sd or tuple(sd[k].shape) != shape:
                    raise ValueError(('Unexpected projector', k))
                projector['backbone.' + k] = list(sd.pop(k).shape)
        if len(sd) != 150:
            raise ValueError(('Expected 150 encoder tensors', len(sd)))
        self.model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.model.load_state_dict(sd, strict=True)
        self.model.eval().requires_grad_(False).float().to(device)
        self.device = torch.device(device)
        self.transform = T.Compose([T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                                    T.CenterCrop(224), T.ToTensor(),
                                    T.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.metadata = dict(model_id=MODEL_ID, revision=REVISION, weight_sha256=digest,
                             provenance='third-party LeJEPA reproduction; encoder-only',
                             excluded_training_projector=projector, strict_encoder_tensors=150,
                             torch_version=torch.__version__, timm_version=timm.__version__,
                             dtype='float32', layer='final LayerNorm, eps=1e-6',
                             spatial_shape=[14, 14, 768], cls_tokens=1, register_tokens=0,
                             preprocessing='RGB PIL bicubic shortedge256 center224 /255 ImageNet mean/std',
                             grid_order='channel-first flatten C,grid_row,grid_col',
                             extra_output_normalization=False)

    def preprocess(self, images):
        """Iterable of PIL images or uint8 HWC RGB arrays -> CPU NCHW float32."""
        from PIL import Image
        return self.torch.stack([self.transform((im if isinstance(im, Image.Image)
                                 else Image.fromarray(im)).convert('RGB')) for im in images])

    def encode(self, batch):
        """Preprocessed tensor -> final tokens, CLS, mean768 and grid3072 on device."""
        torch = self.torch
        with torch.inference_mode():
            z = self.model.forward_features(batch.to(self.device, dtype=torch.float32))
            if tuple(z.shape[1:]) != (197, 768):
                raise ValueError(tuple(z.shape))
            patches = z[:, 1:]
            spatial = patches.transpose(1, 2).reshape(-1, 768, 14, 14)
            grid = torch.nn.functional.adaptive_avg_pool2d(spatial, (2, 2)).flatten(1)
            return dict(tokens=patches, cls=z[:, 0], mean=patches.mean(1), grid=grid)


if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--checkpoint-dir', required=True)
    a.add_argument('--device', default='cpu')
    a.add_argument('--smoke-output', required=True)
    args = a.parse_args()
    import numpy as np
    adapter = FamilyImageAdapter(args.checkpoint_dir, args.device)
    out = adapter.encode(adapter.preprocess([np.zeros((128, 128, 3), dtype=np.uint8)]))
    result = dict(adapter.metadata, smoke={k: dict(shape=list(v.shape),
                  finite=bool(adapter.torch.isfinite(v).all()), mean=float(v.mean())) for k, v in out.items()})
    Path(args.smoke_output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
