"""Adds SSIM column to the component-decomposition ablation (Table 6).

For each backbone × {GradCAM, i-GradCAM, +PCF, CA2_full}: compute the SSIM
between the heatmap on the original image and the heatmap on the rotated image
(after the heatmap is rotated back to align). Averaged over 100 images × 3
angles {15°, 30°, 45°} (matches paper §4 spec).

Output: ./results_imagenet/SSIM_ABLATION.json
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper,
    GradCAM_v2, iGradCAM, iGradCAM_PCF, CA2GradCAM_v2,
    load_imagenet_val, get_model_predictions,
    rpbh_rotate, rotate_heatmap_np, _make_caller,
)

# scikit-image structural_similarity
try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:  # older scikit-image
    from skimage.measure import compare_ssim as ssim_fn

N_IMGS    = 100
EQ_ANGLES = [15.0, 30.0, 45.0]
VAL_DIR   = './imagenet_val'
OUT_PATH  = './results_imagenet/SSIM_ABLATION.json'


def ssim_eq_score(method_fn, imgs, cidxs, angles):
    """Mean SSIM(hm_at_rot_brought_back, hm_original) over (img, angle) pairs."""
    vals = []
    for img, cidx in zip(imgs, cidxs):
        try:
            hm0 = method_fn(img, cidx)
        except Exception:
            continue
        if hm0.std() < 1e-8:
            continue
        for a in angles:
            try:
                hm_r = method_fn(rpbh_rotate(img, float(a)), cidx)
            except Exception:
                continue
            if hm_r.std() < 1e-8:
                continue
            # bring rotated-image heatmap back to original orientation
            ref = rotate_heatmap_np(hm0, float(a))
            try:
                s = ssim_fn(hm_r.astype(np.float32),
                             ref.astype(np.float32),
                             data_range=max(hm_r.max(), ref.max()) - min(hm_r.min(), ref.min()) + 1e-8)
            except Exception:
                continue
            vals.append(float(s))
    return float(np.mean(vals)) if vals else 0.0, float(np.std(vals)) if vals else 0.0


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs_all, labs_all = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    print(f'[DATA] loaded {len(imgs_all)} ImageNet val images', flush=True)
    imgs = imgs_all[:N_IMGS]
    labs = labs_all[:N_IMGS]

    results = {}
    if os.path.exists(OUT_PATH):
        try:
            results = json.load(open(OUT_PATH))
        except Exception:
            results = {}

    def save():
        tmp = OUT_PATH + '.tmp'
        with open(tmp, 'w') as f: json.dump(results, f, indent=2, default=str)
        os.replace(tmp, OUT_PATH)

    for bb in ['resnet50', 'vgg16', 'vit_b_16']:
        if bb in results and isinstance(results[bb], dict) and \
           all(v in results[bb] for v in ['GradCAM', 'i-GradCAM', '+PCF', 'CA2_full']):
            print(f'  [{bb}] SKIP — all 4 variants done', flush=True)
            continue
        print(f'\n=== {bb} ===', flush=True)
        wrapper = BackboneWrapper(bb, device)
        labs_pred = get_model_predictions(wrapper, imgs, labs)

        methods = {
            'GradCAM':   GradCAM_v2(wrapper),
            'i-GradCAM': iGradCAM(wrapper, n_angles=18),
            '+PCF':      iGradCAM_PCF(wrapper, n_angles=18, tau=0.10),
            'CA2_full':  CA2GradCAM_v2(wrapper, tau=0.10, n_angles=18),
        }
        results.setdefault(bb, {})
        for name, m in methods.items():
            if name in results[bb] and 'ssim' in results[bb][name]:
                print(f'  [{bb}/{name}] SKIP', flush=True)
                continue
            print(f'  [{bb}/{name}] computing SSIM on {N_IMGS} imgs × {len(EQ_ANGLES)} angles...', flush=True)
            call = _make_caller(m)
            t0 = time.time()
            ssim_mean, ssim_std = ssim_eq_score(call, imgs, labs_pred, EQ_ANGLES)
            elapsed = time.time() - t0
            results[bb][name] = {
                'ssim':     ssim_mean,
                'ssim_std': ssim_std,
                'n':        N_IMGS,
                'angles':   EQ_ANGLES,
                'time_total': elapsed,
            }
            save()
            print(f'  [{bb}/{name}] SSIM={ssim_mean:.3f}±{ssim_std:.3f}  ({elapsed:.0f}s)', flush=True)
        wrapper.remove_hooks()

    print(f'[SAVED] {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
