"""Job E — PEUM image-level triage validation at n=10,000 on ResNet-50.

Computes:
  - PEUM image-level signal: mean per-pixel weighted variance across views
  - Rotation-instability proxy: 1 - per-image Grad-CAM equivariance
  - Pearson r between the two (overall + top decile of PEUM)
  - Per-pixel Pearson (acknowledged weaker, reported for completeness)

Does NOT modify v11 method/hook/aggregation code. Imports CA2GradCAM_v2 with
return_peum=True (the v1-PEUM by-product flag already in v11).

Output: results_imagenet_official/resnet50__PEUM_validation.json
Walltime target: ~30 min on A100.
"""
import os, sys, json, time
import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper,
    GradCAM_v2, CA2GradCAM_v2,
    load_imagenet_val, get_model_predictions,
    rpbh_rotate, rotate_heatmap_np, _make_caller,
)

VAL_DIR  = './imagenet_val_official'
OUT      = './results_imagenet_official/resnet50__PEUM_validation.json'
N        = 10000          # image-level sample
N_EQ_PROXY_ANGLES = [45.0] # one angle suffices for the proxy; PEUM is rotation-agnostic
BB       = 'resnet50'


def per_image_eq(call, img, cidx, angle=45.0):
    try:
        hm0 = call(img, cidx)
    except Exception:
        return None
    if hm0.std() < 1e-8: return None
    try:
        rot = rpbh_rotate(img, float(angle))
        hm_r = call(rot, cidx)
    except Exception:
        return None
    if hm_r.std() < 1e-8: return None
    ref = rotate_heatmap_np(hm0, float(angle))
    if ref.std() < 1e-8: return None
    try:
        p, _ = pearsonr(hm_r.flatten(), ref.flatten())
    except Exception:
        return None
    return float(p) if np.isfinite(p) else None


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'[DATA] Loading ImageNet val from {VAL_DIR}', flush=True)
    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    print(f'[DATA] {len(imgs)} images', flush=True)
    imgs = imgs[:N]; labs = labs[:N]

    wrapper = BackboneWrapper(BB, device)
    labs = get_model_predictions(wrapper, imgs, labs)

    gc = GradCAM_v2(wrapper)
    ca = CA2GradCAM_v2(wrapper, tau=0.0, n_angles=18)  # tau=0 EquiGrad-CAM headline

    gc_call = _make_caller(gc)

    peum_signal  = []   # image-level mean PEUM variance
    eq_gc_signal = []   # per-image Grad-CAM equivariance (used for 1-Eq proxy)
    pixel_peum_samples = []  # subsampled per-pixel PEUM values
    pixel_disp_samples = []  # corresponding rotation-displacement values

    t0 = time.time()
    for i, (img, cidx) in enumerate(zip(imgs, labs)):
        # Grad-CAM Eq proxy (1-Eq is the rotation-instability target)
        eq_gc = per_image_eq(gc_call, img, cidx, angle=45.0)
        if eq_gc is None:
            continue

        # PEUM signal: from CA2GradCAM with return_peum=True
        try:
            hm, peum = ca(img, cidx, return_peum=True)
        except Exception:
            continue
        if peum is None or peum.std() < 1e-12:
            continue

        peum_mean = float(np.mean(peum))
        peum_signal.append(peum_mean)
        eq_gc_signal.append(eq_gc)

        # Per-pixel sampling: take 100 random pixels per image
        # disp proxy: compare to the rotated heatmap difference
        if i % 5 == 0:
            try:
                hm0 = gc_call(img, cidx)
                rot = rpbh_rotate(img, 45.0)
                hm_r = gc_call(rot, cidx)
                ref = rotate_heatmap_np(hm0, 45.0)
                disp = (hm_r - ref) ** 2  # per-pixel squared displacement
                # Subsample 100 random pixel indices
                H, W = peum.shape
                idx = np.random.choice(H * W, size=min(100, H * W), replace=False)
                pixel_peum_samples.extend(peum.flatten()[idx].tolist())
                pixel_disp_samples.extend(disp.flatten()[idx].tolist())
            except Exception:
                pass

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            r_so_far, _ = pearsonr(peum_signal, [1.0 - eq for eq in eq_gc_signal])
            print(f'  {i+1}/{N}  collected={len(peum_signal)}  r_img={r_so_far:.3f}  '
                  f'elapsed={elapsed/60:.1f}min', flush=True)

    # Image-level correlation
    rot_inst = [1.0 - eq for eq in eq_gc_signal]
    r_img, p_img = pearsonr(peum_signal, rot_inst)

    # Top decile
    arr_peum = np.array(peum_signal)
    arr_inst = np.array(rot_inst)
    thr = np.percentile(arr_peum, 90)
    mask = arr_peum >= thr
    if mask.sum() >= 5:
        r_top, p_top = pearsonr(arr_peum[mask], arr_inst[mask])
    else:
        r_top, p_top = (None, None)

    # Per-pixel (acknowledged weaker)
    if len(pixel_peum_samples) >= 10:
        r_pix, p_pix = pearsonr(pixel_peum_samples, pixel_disp_samples)
    else:
        r_pix, p_pix = (None, None)

    result = {
        'config': {
            'backbone': BB, 'n_target': N, 'n_collected': len(peum_signal),
            'eq_proxy_angle': 45.0, 'tau': 0.0, 'n_views': 18,
        },
        'image_level': {
            'r': float(r_img), 'p': float(p_img), 'n': len(peum_signal),
        },
        'top_decile': {
            'r': float(r_top) if r_top is not None else None,
            'p': float(p_top) if p_top is not None else None,
            'n': int(mask.sum()),
        },
        'per_pixel': {
            'r': float(r_pix) if r_pix is not None else None,
            'p': float(p_pix) if p_pix is not None else None,
            'n_samples': len(pixel_peum_samples),
        },
        'elapsed_min': (time.time() - t0) / 60.0,
    }

    with open(OUT, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'\n[SAVED] {OUT}', flush=True)
    print(f'  image-level Pearson r = {r_img:.4f}   (p={p_img:.2e}, n={len(peum_signal)})', flush=True)
    if r_top is not None:
        print(f'  top-decile  Pearson r = {r_top:.4f}   (p={p_top:.2e}, n={mask.sum()})', flush=True)
    if r_pix is not None:
        print(f'  per-pixel   Pearson r = {r_pix:.4f}   (p={p_pix:.2e}, n={len(pixel_peum_samples)})', flush=True)


if __name__ == '__main__':
    main()
