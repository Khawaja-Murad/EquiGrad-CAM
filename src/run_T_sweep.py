#!/usr/bin/env python3
"""
run_T_sweep.py -- Eq as a function of the number of aggregated views T.
=======================================================================
Practical question: what is the marginal benefit of each additional view? T=6 is
the deployable operating point and T=18 the reference configuration, because
18.7-20.2x a Grad-CAM pass is expensive for an auditing pipeline that processes
large batches. This produces the whole curve so a practitioner can pick a point,
and so the cost/benefit argument is carried by data rather than by two endpoints.

T=1 is NOT included as "Grad-CAM": iGradCAM with n_angles=1 samples the single
view at -180 degrees (np.linspace(-180,180,1,endpoint=False)), which is an aligned
single rotated view, not the single-view baseline. Grad-CAM is measured separately
and reported as the T=0 reference row.

SAMPLING: class-spread (i % 10 < 2), never a prefix.
COST: sum(T) = 54 views x 8 Eq evaluations x n images, plus the Grad-CAM reference.

Output: results_imagenet_official/<backbone>__TSWEEP.json
        atomic save every 25 images; resume via per-T cursors.

Usage: run_T_sweep.py --backbone resnet50 [--n_eq 1000]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, os.path.expanduser('~/scratch/ca2gradcam'))
from ca2_complete_eval import (
    set_seed, BackboneWrapper, GradCAM_v2, iGradCAM, _make_caller,
    load_imagenet_val, get_model_predictions,
    rpbh_rotate, rotate_heatmap_np,
)

VAL_DIR    = os.path.expanduser('~/scratch/ca2gradcam/imagenet_val_official')
OUT_DIR    = os.path.expanduser('~/scratch/ca2gradcam/results_imagenet_official')
EQ_ANGLES  = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]
T_VALUES   = [2, 3, 4, 6, 9, 12, 18]
PER_CLASS  = 2
SAVE_EVERY = 25


def save_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def eq_one(call, img, cidx, angles):
    try:
        hm0 = call(img, cidx)
    except Exception:
        return None
    if hm0.std() < 1e-8:
        return 0.0                      # uniform zero-fill convention
    vals = []
    for a in angles:
        try:
            hm_r = call(rpbh_rotate(img, float(a)), cidx)
        except Exception:
            continue
        if hm_r.std() < 1e-8:
            continue
        ref = rotate_heatmap_np(hm0, float(a))
        if ref.std() < 1e-8:
            continue
        try:
            p, _ = pearsonr(hm_r.flatten(), ref.flatten())
        except Exception:
            continue
        if np.isfinite(p):
            vals.append(float(p))
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backbone', required=True,
                    choices=('resnet50', 'vgg16', 'vit_b_16'))
    ap.add_argument('--n_eq', type=int, default=1000)
    args = ap.parse_args()

    out_path = os.path.join(OUT_DIR, f'{args.backbone}__TSWEEP.json')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(42)
    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    # Class-spread sampling that stays spread AFTER truncation.
    # The old form `[i for i in range(len(imgs)) if (i%10) < PER_CLASS][:n_eq]`
    # silently degenerated into a class PREFIX whenever n_eq < 1000*PER_CLASS
    # (n_eq=1000 with PER_CLASS=2 covered only classes 0-499), which is exactly
    # the sampling bias sec/5 warns about. Order by within-class slot first so
    # truncation removes a whole slot, never a block of classes.
    n_cls = len(imgs) // 10
    cand = [i for i in range(len(imgs)) if (i % 10) < PER_CLASS]
    cand.sort(key=lambda i: (i % 10, i // 10))
    if args.n_eq < n_cls:                    # fewer images than classes:
        step = n_cls / float(args.n_eq)      # take evenly spaced classes
        spread = [int(round(k * step)) * 10 for k in range(args.n_eq)]
        spread = sorted(dict.fromkeys(spread))[:args.n_eq]
    else:
        spread = cand[:args.n_eq]
    print(f'[SAMPLE] class-spread {len(spread)} imgs', flush=True)

    wrapper = BackboneWrapper(args.backbone, device)
    labs = get_model_predictions(wrapper, imgs, labs)

    variants = {'gradcam': _make_caller(GradCAM_v2(wrapper))}
    for T in T_VALUES:
        variants[f'T{T}'] = _make_caller(iGradCAM(wrapper, n_angles=T))

    state = {k: {'eq': [], 'i_next': 0} for k in variants}
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
            if 'state' in old:
                for k in variants:
                    if k in old['state']:
                        state[k] = old['state'][k]
                print('[RESUME] ' + ', '.join(f'{k}@{state[k]["i_next"]}' for k in variants),
                      flush=True)
        except Exception as e:
            print(f'[RESUME] failed: {e}', flush=True)

    def finalize():
        res = {}
        for k in variants:
            a = np.array(state[k]['eq'], float)
            res[k] = {'eq': float(a.mean()) if a.size else None,
                      'eq_std': float(a.std()) if a.size else None,
                      'n': int(a.size),
                      'T': (0 if k == 'gradcam' else int(k[1:]))}
        base = res['gradcam']['eq']
        full = res.get('T18', {}).get('eq')
        if base is not None and full is not None and full != base:
            for k, v in res.items():
                if v['eq'] is not None:
                    v['frac_of_T18_gain'] = float((v['eq'] - base) / (full - base))
        obj = {'config': {'backbone': args.backbone, 'n_eq': len(spread),
                          'T_values': T_VALUES, 'eq_angles': EQ_ANGLES, 'seed': 42,
                          'sampling': f'class-spread {PER_CLASS}/class',
                          'note': 'gradcam row is the single-view T=0 reference; '
                                  'iGradCAM(n_angles=1) would be one aligned rotated '
                                  'view, not the baseline, so T=1 is omitted'},
               'result': res, 'state': state}
        save_atomic(out_path, obj)
        return obj

    t0 = time.time()
    for k, call in variants.items():
        s = state[k]
        print(f'\n[{args.backbone}] {k}: from {s["i_next"]}/{len(spread)}', flush=True)
        for j in range(s['i_next'], len(spread)):
            idx = spread[j]
            e = eq_one(call, imgs[idx], labs[idx], EQ_ANGLES)
            if e is not None:
                s['eq'].append(e)
            s['i_next'] = j + 1
            if (j + 1) % SAVE_EVERY == 0:
                finalize()
                cur = np.mean(s['eq']) if s['eq'] else float('nan')
                print(f'  [{k} {j+1}/{len(spread)}] Eq={cur:.4f}  '
                      f'{(time.time()-t0)/60:.1f}m', flush=True)

    obj = finalize()
    print('\n' + '=' * 60)
    print(f'T SWEEP  {args.backbone}  (n={len(spread)})')
    for k in variants:
        v = obj['result'][k]
        fr = v.get('frac_of_T18_gain')
        fr_s = f'{fr:6.1%} of T=18 gain' if fr is not None else ''
        print(f'  T={v["T"]:<3d} Eq={v["eq"]:.4f}  n={v["n"]:<5d} {fr_s}')
    print(f'[SAVED] {out_path}', flush=True)
    wrapper.remove_hooks()


if __name__ == '__main__':
    main()
