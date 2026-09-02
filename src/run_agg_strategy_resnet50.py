"""Aggregation strategy ablation (Table 7) — ResNet-50 / ImageNet-1K.

Compare 4 ways to aggregate over rotated views:
  - pixel-space    (AugCAM τ=0): aggregate upsampled heatmaps in pixel space
  - per-view ReLU  at h×w:       relu(α A) per view, average at feature scale
  - per-view α A   at h×w:       α A per view, average at feature scale
  - proposed       (EquiGrad-CAM): average A then α, combine, ReLU once

Output: ./results_imagenet/AGG_STRATEGY_resnet50.json
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper, AugCAM_v2,
    load_imagenet_val, get_model_predictions,
    insertion_deletion, eq_scores, _make_caller,
    rpbh_rotate, inv_rotate, make_heatmap_v2,
)

N_IMGS    = 500
N_EQ      = 500
N_ANGLES  = 18
TAU       = 0.10
EQ_ANGLES = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]
VAL_DIR   = './imagenet_val'
OUT_PATH  = './results_imagenet/AGG_STRATEGY_resnet50.json'


class AggPerViewReLU:
    """Per-view: relu(sum_k mean(g_k) * A_k), interpolate, then average over views."""
    def __init__(self, wrapper, n_angles=18, tau=TAU, pad=64):
        self.w = wrapper; self.n = n_angles; self.tau = tau; self.pad = pad
    def __call__(self, img, cidx):
        self.w.model.eval()
        H, W = img.shape[1], img.shape[2]
        per_view = []
        for a in np.linspace(-180, 180, self.n, endpoint=False):
            rot = rpbh_rotate(img, float(a), self.pad)
            A, g, conf, pred = self.w.gradcam_pass(rot.unsqueeze(0), cidx)
            if pred != cidx or conf < self.tau:
                continue
            hm = make_heatmap_v2(inv_rotate(A, a), inv_rotate(g, a), (H, W), relu_weights=self.w.is_vit)
            per_view.append(hm.numpy())
        if not per_view:
            return np.zeros((H, W), dtype=np.float32)
        return np.mean(per_view, axis=0)


class AggPerViewAlphaA:
    """Per-view: (mean_ij g_kij) * A_k summed over k, NO per-view relu, interp, average."""
    def __init__(self, wrapper, n_angles=18, tau=TAU, pad=64):
        self.w = wrapper; self.n = n_angles; self.tau = tau; self.pad = pad
    def __call__(self, img, cidx):
        self.w.model.eval()
        H, W = img.shape[1], img.shape[2]
        per_view = []
        for a in np.linspace(-180, 180, self.n, endpoint=False):
            rot = rpbh_rotate(img, float(a), self.pad)
            A, g, conf, pred = self.w.gradcam_pass(rot.unsqueeze(0), cidx)
            if pred != cidx or conf < self.tau:
                continue
            Ai = inv_rotate(A, a)
            gi = inv_rotate(g, a)
            alpha = gi.mean(dim=(1, 2))
            cam = (alpha[:, None, None] * Ai).sum(0)
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (H, W),
                                  mode='bilinear', align_corners=False).squeeze()
            per_view.append(cam.numpy())
        if not per_view:
            return np.zeros((H, W), dtype=np.float32)
        out = np.mean(per_view, axis=0)
        out = np.maximum(out, 0)
        mn, mx = out.min(), out.max()
        if mx > mn:
            out = (out - mn) / (mx - mn)
        return out


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    print(f'[DATA] {len(imgs)} images', flush=True)
    imgs = imgs[:N_IMGS]
    labs = labs[:N_IMGS]

    wrapper = BackboneWrapper('resnet50', device)
    labs = get_model_predictions(wrapper, imgs, labs)

    # CA2GradCAM_v2 has the proposed avg(A), avg(g) strategy.
    from ca2_complete_eval import CA2GradCAM_v2

    strategies = {
        'pixel_space_AugCAM_t0':     AugCAM_v2(wrapper, tau=0.0),
        'per_view_ReLU_at_hw':       AggPerViewReLU(wrapper, n_angles=N_ANGLES, tau=TAU),
        'per_view_alphaA_at_hw':     AggPerViewAlphaA(wrapper, n_angles=N_ANGLES, tau=TAU),
        'proposed_avg_A_avg_g':      CA2GradCAM_v2(wrapper, tau=TAU, n_angles=N_ANGLES),
    }

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

    for name, method in strategies.items():
        if name in results and 'eq' in results[name]:
            print(f'  [{name}] SKIP', flush=True)
            continue
        print(f'\n  [{name}] {len(imgs)} ins/del images, {N_EQ} eq images', flush=True)
        call = _make_caller(method)
        ins_list, del_list = [], []
        t0 = time.time()
        for i, (img, cidx) in enumerate(zip(imgs, labs)):
            try:
                hm = call(img, cidx)
            except Exception as e:
                print(f'    image {i} error: {e}', flush=True)
                continue
            ii, dd = insertion_deletion(wrapper, img, hm, cidx, 20, 10)
            ins_list.append(ii); del_list.append(dd)
            if (i+1) % 100 == 0:
                print(f'    {i+1}/{len(imgs)}  Ins={np.mean(ins_list):.3f}', flush=True)
        print(f'    Equivariance ({N_EQ} imgs)...', flush=True)
        eq_dict = eq_scores(call, imgs[:N_EQ], labs[:N_EQ], EQ_ANGLES)
        elapsed = time.time() - t0
        results[name] = {
            'n':        len(imgs),
            'n_eq':     N_EQ,
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
        r = results[name]
        print(f'  [{name}] Eq={r["eq"]:.3f}±{r["eq_std"]:.3f} Ins={r["ins"]:.3f} Del={r["del"]:.3f}', flush=True)

    print(f'[SAVED] {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
