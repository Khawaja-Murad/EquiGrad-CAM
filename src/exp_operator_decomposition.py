#!/usr/bin/env python3
"""
exp_operator_decomposition.py -- WHERE does rotation drift enter the CAM operator?
==================================================================================
Motivation. Arguing that Grad-CAM's rotation drift is "operator noise" from
penultimate FEATURE and LOGIT cosine similarity is not sufficient, for an exact
reason:

    Grad-CAM is not computed from the penultimate feature or the logit vector.
    It is computed from the class-specific GRADIENT field and the spatial
    activations. A network can hold its logits and its top-1 fixed while its
    local gradient field moves.

So we measure the quantities that actually enter the operator. make_heatmap_v2
(ca2_complete_eval.py:197) is exactly:

    alpha = grads.mean(dim=(1,2))      # channel weights, from ONE orientation
    cam   = (alpha . acts).sum(0)      # weighted sum
    cam   = ReLU(cam)                  # rectification
    cam   = interpolate(cam)           # upsample
    cam   = (cam - min) / (max - min)  # min-max normalise

We report equivariance at EVERY stage of that chain, so the drift can be
attributed to a specific step rather than inferred.

CONVENTION. We match the paper's Eq definition exactly: the comparison happens in
the ROTATED frame, i.e. we compare  q(R_theta x)  against  R_theta q(x)  for each
intermediate q, using the FROZEN inv_rotate for the forward action (rot_feat below)
and the FROZEN rotate_heatmap_np at the final 224x224 stage. Nothing here
re-defines the metric; it applies the paper's own metric further up the pipeline.

STAGES REPORTED (Pearson unless noted)
  eq_acts       activations A            -- do conv features rotate with the input?
  eq_grads      gradient field g         -- the quantity that actually drives the CAM
  eq_alpha      channel weights (cosine AND Pearson; a C-vector, not spatial, so
                no alignment applies -- GAP is rotation-invariant in the ideal case)
  eq_pre_relu   (alpha . A).sum(0) before ReLU
  eq_post_relu  after ReLU
  eq_final      full heatmap == the Eq the paper reports (sanity anchor)

ALTERNATIVE SIMILARITIES at the final stage: Spearman, cosine, SSIM. This answers,
at no extra model cost, the separate question of whether the headline metric is
Pearson-specific and that the constant-map Eq=0 convention drives the ranking.

ALSO LOGGED per (image, angle): the model's predicted class and confidence on the
rotated view, so the target-class protocol question (fixed-original-class vs
per-view-predicted vs prediction-stable-only) is answerable from this same run
without new compute.

INTERPOLATION CAVEAT (reported, not hidden): 90 and 180 degrees are exact lattice
rotations; 15/30/45/60/135 resample a coarse feature grid (7x7 on ResNet-50,
14x14 on VGG-16/ViT), so some of eq_acts/eq_grads degradation at those angles is
resampling, not the network. Per-angle values are stored so this is visible, and
the 90/180 columns give an interpolation-free read.

SAMPLING: class-spread (i % 10 < 2), NOT a prefix -- the loader is class-blocked,
and the paper documents that a contiguous prefix inflates Eq by +0.07..+0.09.

Output: results_imagenet_official/<backbone>__OPDECOMP.json
        atomic save every 50 images; resume via i_next.

Usage: exp_operator_decomposition.py --backbone resnet50 [--n_eq 2000]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, os.path.expanduser('~/scratch/ca2gradcam'))
from ca2_complete_eval import (
    set_seed, BackboneWrapper, make_heatmap_v2,
    load_imagenet_val, get_model_predictions,
    rpbh_rotate, inv_rotate, rotate_heatmap_np,
)

VAL_DIR    = os.path.expanduser('~/scratch/ca2gradcam/imagenet_val_official')
OUT_DIR    = os.path.expanduser('~/scratch/ca2gradcam/results_imagenet_official')
EQ_ANGLES  = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]
PER_CLASS  = 2
SAVE_EVERY = 50
EPS        = 1e-8

STAGES = ['acts', 'grads', 'alpha', 'pre_relu', 'post_relu', 'final']


def rot_feat(t, angle):
    """Forward spatial action R_theta on a (C,H,W) tensor, via the FROZEN operator.

    inv_rotate(t, a) rotates by -a, so the forward action is inv_rotate(t, -a).
    """
    return inv_rotate(t, -angle)


def rowwise_pearson(P, Q):
    """Mean Pearson r over channels. P, Q are (C, H, W) torch tensors.

    Vectorised: a scipy loop over 2048 channels per image-angle would dominate
    runtime. Channels that are constant in either tensor are skipped (r undefined).
    """
    a = P.reshape(P.shape[0], -1).double().numpy()
    b = Q.reshape(Q.shape[0], -1).double().numpy()
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    na = np.sqrt((a * a).sum(1))
    nb = np.sqrt((b * b).sum(1))
    ok = (na > EPS) & (nb > EPS)
    if not ok.any():
        return None
    r = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
    r = r[np.isfinite(r)]
    return float(r.mean()) if r.size else None


def flat_pearson(u, v):
    u = np.asarray(u, float).ravel(); v = np.asarray(v, float).ravel()
    if u.std() < EPS or v.std() < EPS:
        return None
    r, _ = pearsonr(u, v)
    return float(r) if np.isfinite(r) else None


def cosine(u, v):
    u = np.asarray(u, float).ravel(); v = np.asarray(v, float).ravel()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < EPS or nv < EPS:
        return None
    return float(np.dot(u, v) / (nu * nv))


def save_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backbone', required=True,
                    choices=('resnet50', 'vgg16', 'vit_b_16'))
    ap.add_argument('--n_eq', type=int, default=2000)
    args = ap.parse_args()

    out_path = os.path.join(OUT_DIR, f'{args.backbone}__OPDECOMP.json')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(42)
    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    spread = [i for i in range(len(imgs)) if (i % 10) < PER_CLASS][:args.n_eq]
    print(f'[SAMPLE] class-spread {len(spread)} imgs '
          f'({PER_CLASS}/class over {len(spread)//PER_CLASS} classes)', flush=True)

    wrapper = BackboneWrapper(args.backbone, device)
    labs = get_model_predictions(wrapper, imgs, labs)
    is_vit = wrapper.is_vit

    # running accumulators: stage -> angle -> list of per-image values
    def new_acc():
        return {s: {str(a): [] for a in EQ_ANGLES} for s in STAGES}
    state = {
        'pearson': new_acc(),
        'alpha_cos': {str(a): [] for a in EQ_ANGLES},
        'final_spearman': {str(a): [] for a in EQ_ANGLES},
        'final_cosine': {str(a): [] for a in EQ_ANGLES},
        'final_ssim': {str(a): [] for a in EQ_ANGLES},
        'pred_stable': {str(a): [] for a in EQ_ANGLES},   # target-class protocol
        'conf_ratio': {str(a): [] for a in EQ_ANGLES},
        'i_next': 0,
    }
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
            if 'state' in old:
                state = old['state']
                print(f'[RESUME] from image {state["i_next"]}', flush=True)
        except Exception as e:
            print(f'[RESUME] failed: {e}', flush=True)

    def finalize():
        res = {}
        for s in STAGES:
            pa = {a: (float(np.mean(v)) if v else None)
                  for a, v in state['pearson'][s].items()}
            vals = [v for v in pa.values() if v is not None]
            res[s] = {'per_angle': pa,
                      'mean': float(np.mean(vals)) if vals else None,
                      'n_per_angle': {a: len(v) for a, v in state['pearson'][s].items()}}
        def agg(d):
            pa = {a: (float(np.mean(v)) if v else None) for a, v in d.items()}
            vals = [v for v in pa.values() if v is not None]
            return {'per_angle': pa, 'mean': float(np.mean(vals)) if vals else None}
        obj = {
            'config': {'backbone': args.backbone, 'n_eq': len(spread),
                       'sampling': f'class-spread {PER_CLASS}/class',
                       'eq_angles': EQ_ANGLES, 'seed': 42,
                       'convention': 'compare q(R x) vs R q(x) in the rotated frame '
                                     '(matches the papers Eq definition)',
                       'exact_angles_no_interpolation': [90.0, 180.0]},
            'stages_pearson': res,
            'alpha_cosine': agg(state['alpha_cos']),
            'final_alt_similarity': {
                'spearman': agg(state['final_spearman']),
                'cosine':   agg(state['final_cosine']),
                'ssim':     agg(state['final_ssim']),
            },
            'target_class_protocol': {
                'pred_stable_rate': agg(state['pred_stable']),
                'conf_ratio': agg(state['conf_ratio']),
            },
            'state': state,
        }
        save_atomic(out_path, obj)
        return obj

    t0 = time.time()
    for k in range(state['i_next'], len(spread)):
        idx = spread[k]
        img, c = imgs[idx], labs[idx]
        try:
            A0, g0, conf0, pred0 = wrapper.gradcam_pass(img.unsqueeze(0), c)
        except Exception:
            state['i_next'] = k + 1
            continue
        alpha0 = g0.mean(dim=(1, 2))
        pre0 = (alpha0[:, None, None] * A0).sum(0)
        h0 = make_heatmap_v2(A0, g0, (img.shape[1], img.shape[2]),
                             relu_weights=is_vit).numpy()

        for a in EQ_ANGLES:
            ang = float(a); key = str(a)
            try:
                rot = rpbh_rotate(img, ang)
                At, gt, conft, predt = wrapper.gradcam_pass(rot.unsqueeze(0), c)
            except Exception:
                continue

            # --- forward-rotate the 0-degree side into the rotated frame ---
            A0r = rot_feat(A0, ang)
            g0r = rot_feat(g0, ang)
            pre0r = rot_feat(pre0.unsqueeze(0), ang).squeeze(0)

            v = rowwise_pearson(At, A0r)
            if v is not None: state['pearson']['acts'][key].append(v)
            v = rowwise_pearson(gt, g0r)
            if v is not None: state['pearson']['grads'][key].append(v)

            # alpha is a channel vector: GAP is rotation-invariant in the ideal
            # case, so it is compared directly with no spatial alignment.
            alphat = gt.mean(dim=(1, 2))
            v = flat_pearson(alphat.numpy(), alpha0.numpy())
            if v is not None: state['pearson']['alpha'][key].append(v)
            v = cosine(alphat.numpy(), alpha0.numpy())
            if v is not None: state['alpha_cos'][key].append(v)

            pret = (alphat[:, None, None] * At).sum(0)
            v = flat_pearson(pret.numpy(), pre0r.numpy())
            if v is not None: state['pearson']['pre_relu'][key].append(v)
            v = flat_pearson(torch.relu(pret).numpy(), torch.relu(pre0r).numpy())
            if v is not None: state['pearson']['post_relu'][key].append(v)

            # --- final stage: the paper's own Eq, plus alternative similarities ---
            ht = make_heatmap_v2(At, gt, (img.shape[1], img.shape[2]),
                                 relu_weights=is_vit).numpy()
            ref = rotate_heatmap_np(h0, ang)
            if ht.std() > EPS and ref.std() > EPS:
                v = flat_pearson(ht, ref)
                if v is not None: state['pearson']['final'][key].append(v)
                try:
                    sr, _ = spearmanr(ht.ravel(), ref.ravel())
                    if np.isfinite(sr): state['final_spearman'][key].append(float(sr))
                except Exception:
                    pass
                v = cosine(ht, ref)
                if v is not None: state['final_cosine'][key].append(v)
                try:
                    dr = float(max(ht.max(), ref.max()) - min(ht.min(), ref.min()))
                    if dr > EPS:
                        state['final_ssim'][key].append(
                            float(ssim_fn(ht, ref, data_range=dr)))
                except Exception:
                    pass

            state['pred_stable'][key].append(1.0 if predt == pred0 else 0.0)
            state['conf_ratio'][key].append(float(conft / max(conf0, EPS)))

        state['i_next'] = k + 1
        if (k + 1) % SAVE_EVERY == 0:
            finalize()
            def m(s):
                v = state['pearson'][s]['135.0']
                return np.mean(v) if v else float('nan')
            print(f'  [{k+1}/{len(spread)}] @135deg  A={m("acts"):.3f} g={m("grads"):.3f} '
                  f'alpha={m("alpha"):.3f} pre={m("pre_relu"):.3f} '
                  f'final={m("final"):.3f}   {(time.time()-t0)/60:.1f}m', flush=True)

    obj = finalize()
    print('\n' + '=' * 84)
    print(f'OPERATOR DECOMPOSITION  {args.backbone}  (n={len(spread)})')
    print(f'  {"stage":11s} ' + ' '.join(f'{str(a).replace(".0",""):>7s}' for a in EQ_ANGLES) + '    mean')
    for s in STAGES:
        r = obj['stages_pearson'][s]
        row = ' '.join(f'{(r["per_angle"][str(a)] if r["per_angle"][str(a)] is not None else float("nan")):7.3f}'
                       for a in EQ_ANGLES)
        mn = r['mean'] if r['mean'] is not None else float('nan')
        print(f'  {s:11s} {row}  {mn:7.3f}')
    print(f'\n  alpha cosine mean : {obj["alpha_cosine"]["mean"]}')
    alt = obj['final_alt_similarity']
    print(f'  final Pearson     : {obj["stages_pearson"]["final"]["mean"]}')
    print(f'  final Spearman    : {alt["spearman"]["mean"]}')
    print(f'  final cosine      : {alt["cosine"]["mean"]}')
    print(f'  final SSIM        : {alt["ssim"]["mean"]}')
    print(f'[SAVED] {out_path}', flush=True)
    wrapper.remove_hooks()


if __name__ == '__main__':
    main()
