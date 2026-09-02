"""ViT hook ablation (§6.4 of the paper).

Hook the forward+backward at three locations in the last encoder block:
  - ln_1            (current / paper choice — pre-attention LayerNorm)
  - ln_2            (pre-MLP LayerNorm)
  - full block out  (post-residual output of encoder.layers[-1])

For each, compute EquiGrad-CAM N=18 Eq on a 500-image ImageNet-1K subset
and also report the *constancy* of the heatmap (heatmap.std() over a sample).
Paper claims only ln_1 produces non-constant heatmaps because the residual
connection distributes gradients uniformly across the other two locations.

Output: ./results_imagenet/VIT_HOOK_ABL.json
"""
import os, sys, json, time
import numpy as np
import torch
import torchvision.models as models

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper, CA2GradCAM_v2,
    load_imagenet_val, get_model_predictions,
    insertion_deletion, eq_scores, _make_caller,
)

N_IMGS    = 500    # ins/del subset
N_EQ      = 500    # equivariance subset
EQ_ANGLES = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]
VAL_DIR   = './imagenet_val'
OUT_PATH  = './results_imagenet/VIT_HOOK_ABL.json'


class HookSwapViTWrapper(BackboneWrapper):
    """ViT wrapper that hooks the user-specified location in the last block."""
    def __init__(self, hook_location, device='cuda'):
        # Mimic BackboneWrapper.__init__ for vit_b_16 but pick the hook layer.
        self.device = device
        self.name = 'vit_b_16'
        self.is_vit = True
        self.spatial_size = (14, 14)
        self._acts, self._grads = [], []
        self.hook_location = hook_location

        self.model = models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()
        blk = self.model.encoder.layers[-1]

        if hook_location == 'ln_1':
            layer = blk.ln_1
        elif hook_location == 'ln_2':
            layer = blk.ln_2
        elif hook_location == 'block':
            layer = blk
        else:
            raise ValueError(hook_location)

        self._fwd = layer.register_forward_hook(self._hook_fwd)
        self._bwd = layer.register_full_backward_hook(self._hook_bwd)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    print(f'[DATA] {len(imgs)} images', flush=True)
    imgs = imgs[:N_IMGS]
    labs = labs[:N_IMGS]

    results = {}
    if os.path.exists(OUT_PATH):
        try:
            results = json.load(open(OUT_PATH))
            done = [k for k in results if 'eq' in (results.get(k) or {})]
            print(f'[RESUME] done: {done}', flush=True)
        except Exception:
            results = {}

    def save():
        tmp = OUT_PATH + '.tmp'
        with open(tmp, 'w') as f: json.dump(results, f, indent=2, default=str)
        os.replace(tmp, OUT_PATH)

    for hook in ['ln_1', 'ln_2', 'block']:
        if hook in results and 'eq' in results[hook]:
            print(f'  [{hook}] SKIP', flush=True)
            continue
        print(f'\n  [{hook}] hooking encoder.layers[-1].{hook}', flush=True)
        try:
            wrapper = HookSwapViTWrapper(hook, device)
        except Exception as e:
            print(f'  [{hook}] wrapper failed: {e}', flush=True)
            continue
        labs_pred = get_model_predictions(wrapper, imgs, labs)

        method = CA2GradCAM_v2(wrapper, tau=0.10, n_angles=18)
        call = _make_caller(method)

        ins_list, del_list, hm_stds = [], [], []
        t0 = time.time()
        for i, (img, cidx) in enumerate(zip(imgs, labs_pred)):
            try:
                hm = call(img, cidx)
            except Exception as e:
                print(f'    image {i} error: {e}', flush=True)
                continue
            hm_stds.append(float(hm.std()))
            ii, dd = insertion_deletion(wrapper, img, hm, cidx, 20, 10)
            ins_list.append(ii); del_list.append(dd)
            if (i+1) % 100 == 0:
                print(f'    {i+1}/{len(imgs)}  hm_std={np.mean(hm_stds):.4f}  Ins={np.mean(ins_list):.3f}', flush=True)
        print(f'    Equivariance ({N_EQ} imgs)...', flush=True)
        eq_dict = eq_scores(call, imgs[:N_EQ], labs_pred[:N_EQ], EQ_ANGLES)
        elapsed = time.time() - t0
        results[hook] = {
            'hook':           hook,
            'n':              len(imgs),
            'n_eq':           N_EQ,
            'heatmap_std_mean': float(np.mean(hm_stds)) if hm_stds else 0.0,
            'heatmap_std_max':  float(np.max(hm_stds))  if hm_stds else 0.0,
            'fraction_constant': float(np.mean(np.array(hm_stds) < 1e-4)) if hm_stds else 1.0,
            'eq':       float(eq_dict['pearson']),
            'eq_std':   float(eq_dict['pearson_std']),
            'ins':      float(np.mean(ins_list)),
            'ins_std':  float(np.std(ins_list)),
            'del':      float(np.mean(del_list)),
            'del_std':  float(np.std(del_list)),
            'time_img': elapsed / max(1, len(imgs)),
            'per_angle': eq_dict['per_angle'],
        }
        save()
        r = results[hook]
        print(f'  [{hook}] Eq={r["eq"]:.3f}  hm_std_mean={r["heatmap_std_mean"]:.4f}  '
              f'frac_constant={r["fraction_constant"]:.2%}', flush=True)
        wrapper.remove_hooks()

    print(f'[SAVED] {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
