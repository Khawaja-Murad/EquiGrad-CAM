"""Rerun the component-decomposition ablation at n=1000 (was n=100 in v2).

For each backbone × {GradCAM, i-GradCAM, +PCF, CA2_full}: compute Eq on 1000
images × 7 angles, Ins on 1000 images, SSIM on 1000 images × 3 angles {15,30,45}°.

Invocation:
  python run_ablation_n1000.py <backbone>

Output: results_imagenet_official/<backbone>__ABL_N1000.json
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper,
    GradCAM_v2, iGradCAM, iGradCAM_PCF, CA2GradCAM_v2,
    load_imagenet_val, get_model_predictions,
    insertion_deletion, eq_scores, _make_caller,
    rpbh_rotate, rotate_heatmap_np,
)
from skimage.metrics import structural_similarity as ssim_fn

VAL_DIR    = './imagenet_val_official'
N          = 1000
EQ_ANGLES  = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]
SSIM_ANGLES = [15.0, 30.0, 45.0]


def ssim_eq_score(call, imgs, cidxs, angles):
    vals = []
    for img, cidx in zip(imgs, cidxs):
        try:
            hm0 = call(img, cidx)
        except Exception:
            continue
        if hm0.std() < 1e-8:
            continue
        for a in angles:
            try:
                hm_r = call(rpbh_rotate(img, float(a)), cidx)
            except Exception:
                continue
            if hm_r.std() < 1e-8:
                continue
            ref = rotate_heatmap_np(hm0, float(a))
            try:
                s = ssim_fn(hm_r.astype(np.float32),
                             ref.astype(np.float32),
                             data_range=max(hm_r.max(), ref.max()) - min(hm_r.min(), ref.min()) + 1e-8)
            except Exception:
                continue
            vals.append(float(s))
    return (float(np.mean(vals)) if vals else 0.0,
            float(np.std(vals)) if vals else 0.0)


def main():
    if len(sys.argv) < 2:
        print('usage: run_ablation_n1000.py <backbone>')
        sys.exit(2)
    bb = sys.argv[1]
    assert bb in ('resnet50', 'vgg16', 'vit_b_16')
    out_path = f'./results_imagenet_official/{bb}__ABL_N1000.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs_all, labs_all = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    imgs = imgs_all[:N]
    labs = labs_all[:N]
    print(f'[DATA] using first {N} of {len(imgs_all)} images', flush=True)

    wrapper = BackboneWrapper(bb, device)
    labs = get_model_predictions(wrapper, imgs, labs)

    methods = {
        'GradCAM':   GradCAM_v2(wrapper),
        'i-GradCAM': iGradCAM(wrapper, n_angles=18),
        '+PCF':      iGradCAM_PCF(wrapper, n_angles=18, tau=0.10),
        'CA2_full':  CA2GradCAM_v2(wrapper, tau=0.10, n_angles=18),
    }

    results = {}
    if os.path.exists(out_path):
        try: results = json.load(open(out_path))
        except: results = {}

    def save():
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f: json.dump(results, f, indent=2, default=str)
        os.replace(tmp, out_path)

    for name, m in methods.items():
        if name in results and 'eq' in (results[name] or {}):
            print(f'  [{name}] SKIP', flush=True)
            continue
        print(f'\n  [{name}] Eq + Ins + SSIM @ n={N}', flush=True)
        call = _make_caller(m)
        t0 = time.time()
        # Eq + Ins
        ins_list = []
        for i, (img, cidx) in enumerate(zip(imgs, labs)):
            try:
                hm = call(img, cidx)
            except Exception:
                continue
            ii, _ = insertion_deletion(wrapper, img, hm, cidx, 20, 10)
            ins_list.append(ii)
            if (i+1) % 200 == 0:
                print(f'    {i+1}/{N}  Ins={np.mean(ins_list):.3f}', flush=True)
        eq = eq_scores(call, imgs, labs, EQ_ANGLES)
        # SSIM
        print(f'    SSIM ...', flush=True)
        ss, ss_std = ssim_eq_score(call, imgs, labs, SSIM_ANGLES)
        elapsed = time.time() - t0
        results[name] = {
            'n':        N,
            'eq':       float(eq['pearson']),
            'eq_std':   float(eq['pearson_std']),
            'ssim':     ss,
            'ssim_std': ss_std,
            'ins':      float(np.mean(ins_list)) if ins_list else 0.0,
            'ins_std':  float(np.std(ins_list)) if ins_list else 0.0,
            'time_total': elapsed,
        }
        save()
        r = results[name]
        print(f'  [{name}] Eq={r["eq"]:.3f} SSIM={r["ssim"]:.3f} Ins={r["ins"]:.3f}  ({elapsed/60:.1f} min)', flush=True)

    print(f'\n[SAVED] {out_path}', flush=True)


if __name__ == '__main__':
    main()
