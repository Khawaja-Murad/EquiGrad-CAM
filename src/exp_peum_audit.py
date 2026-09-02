#!/usr/bin/env python3
"""
exp_peum_audit.py -- turn PEUM from a correlation into an audited triage signal.
================================================================================
All three reviews attack the PEUM claim on the same three grounds:

  (a) CIRCULARITY. The ranking and the lift are computed on the SAME images, and
      the "worst decile" target is defined using the SAME rotation family that
      PEUM's variance is computed over.
  (b) NO BASELINE. PEUM is never compared against cheaper signals an auditor
      already has: confidence change under rotation, whether the prediction flips,
      or disagreement among single-view CAMs.
  (c) WRONG TARGET. PEUM is validated as an EXPLANATION-INSTABILITY detector, but
      sold as an AUDIT signal. The sharper test: does it predict actual MODEL
      failure -- a misclassification -- not merely an unstable heatmap?

This runner does NOT compute statistics. It stores the raw per-image quantities so
that the held-out split, bootstrap intervals, lift curves and baseline comparisons
can all be recomputed on CPU without re-running the GPU pass, and so a reader can
audit them. analyze_peum_audit.py does the statistics.

PER IMAGE it records:
  peum            image-level mean view-variance of the T aligned single-view maps
  instability     1 - Eq(Grad-CAM), the explanation-instability target
  correct         model top-1 == dataset label   <- (c), the real-failure target
  conf0           confidence on the unrotated input
  conf_drop_mean  mean (1 - conf_theta/conf_0) over the evaluation angles
  flip_rate       fraction of angles whose top-1 differs from the unrotated top-1
  cam_disagree    mean pairwise (1 - Pearson) among the aligned single-view maps
                  -- a cheap CAM-only competitor to PEUM
  eq_gradcam      raw Eq for reference

conf_drop_mean, flip_rate and cam_disagree are the (b) baselines PEUM must beat.
correct is the (c) target. Held-out splitting and CIs are (a), handled downstream.

SAMPLING: class-spread (i % 10 < 2), never a prefix.
Output: results_imagenet_official/<backbone>__PEUM_AUDIT.json  (per-image arrays)
        atomic save every 50 images; resume via i_next.

Usage: exp_peum_audit.py --backbone resnet50 [--n_eq 2000] [--T 18]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, os.path.expanduser('~/scratch/ca2gradcam'))
from ca2_complete_eval import (
    set_seed, BackboneWrapper, GradCAM_v2, _make_caller,
    load_imagenet_val, get_model_predictions,
    rpbh_rotate, inv_rotate, rotate_heatmap_np,
)

VAL_DIR    = os.path.expanduser('~/scratch/ca2gradcam/imagenet_val_official')
OUT_DIR    = os.path.expanduser('~/scratch/ca2gradcam/results_imagenet_official')
EQ_ANGLES  = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]
PER_CLASS  = 2
SAVE_EVERY = 50
FIELDS = ['peum', 'instability', 'eq_gradcam', 'correct', 'conf0',
          'conf_drop_mean', 'flip_rate', 'cam_disagree']


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
    ap.add_argument('--T', type=int, default=18)
    args = ap.parse_args()

    out_path = os.path.join(OUT_DIR, f'{args.backbone}__PEUM_AUDIT.json')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seed(42)
    imgs, true_labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    spread = [i for i in range(len(imgs)) if (i % 10) < PER_CLASS][:args.n_eq]
    print(f'[SAMPLE] class-spread {len(spread)} imgs, T={args.T}', flush=True)

    wrapper = BackboneWrapper(args.backbone, device)
    # keep BOTH: dataset label (ground truth) and model top-1 (explanation target)
    pred_labs = get_model_predictions(wrapper, imgs, list(true_labs))
    gc = _make_caller(GradCAM_v2(wrapper))
    view_angles = list(np.linspace(-180, 180, args.T, endpoint=False))

    state = {f: [] for f in FIELDS}
    state['i_next'] = 0
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
            if 'per_image' in old and 'i_next' in old:
                for f in FIELDS:
                    state[f] = old['per_image'].get(f, [])
                state['i_next'] = old['i_next']
                print(f'[RESUME] from image {state["i_next"]}', flush=True)
        except Exception as e:
            print(f'[RESUME] failed: {e}', flush=True)

    def finalize():
        obj = {'config': {'backbone': args.backbone, 'n_eq': len(spread), 'T': args.T,
                          'eq_angles': EQ_ANGLES, 'seed': 42,
                          'sampling': f'class-spread {PER_CLASS}/class',
                          'note': 'raw per-image values only; statistics are computed '
                                  'downstream by analyze_peum_audit.py so the held-out '
                                  'split and bootstrap can be redone without a GPU'},
               'fields': {
                   'peum': 'image-level mean variance of the T aligned single-view maps',
                   'instability': '1 - Eq(Grad-CAM) over the 7 evaluation angles',
                   'correct': 'model top-1 == dataset label (real-failure target)',
                   'conf_drop_mean': 'mean 1 - conf_theta/conf_0 (baseline signal)',
                   'flip_rate': 'fraction of angles whose top-1 flips (baseline signal)',
                   'cam_disagree': 'mean pairwise 1-Pearson among aligned views (baseline)',
               },
               'n_records': len(state['peum']),
               'per_image': {f: state[f] for f in FIELDS},
               'i_next': state['i_next']}
        save_atomic(out_path, obj)
        return obj

    t0 = time.time()
    for k in range(state['i_next'], len(spread)):
        idx = spread[k]
        img = imgs[idx]
        c = pred_labs[idx]                       # explain the model's own top-1
        try:
            h0 = gc(img, c)
            if h0.std() < 1e-8:
                state['i_next'] = k + 1; continue
            conf0 = wrapper.score(img, c)

            # ---- T aligned single-view maps -> PEUM and CAM disagreement ----
            views = []
            for a in view_angles:
                try:
                    hm = gc(rpbh_rotate(img, float(a)), c)
                    al = inv_rotate(torch.from_numpy(hm).unsqueeze(0),
                                    float(a)).squeeze(0).numpy()
                    views.append(al)
                except Exception:
                    continue
            if len(views) < 2:
                state['i_next'] = k + 1; continue
            V = np.stack(views, 0)
            peum = float(V.var(axis=0).mean())

            m = min(len(views), 6)               # cap pairs: O(m^2) correlations
            ds = []
            for i in range(m):
                for j in range(i + 1, m):
                    if V[i].std() > 1e-8 and V[j].std() > 1e-8:
                        r, _ = pearsonr(V[i].ravel(), V[j].ravel())
                        if np.isfinite(r):
                            ds.append(1.0 - float(r))
            cam_disagree = float(np.mean(ds)) if ds else float('nan')

            # ---- Eq(Grad-CAM), confidence drop, prediction flips ----
            eqs, drops, flips = [], [], []
            for a in EQ_ANGLES:
                try:
                    rot = rpbh_rotate(img, float(a))
                    hm_r = gc(rot, c)
                    ref = rotate_heatmap_np(h0, float(a))
                    if hm_r.std() > 1e-8 and ref.std() > 1e-8:
                        r, _ = pearsonr(hm_r.ravel(), ref.ravel())
                        if np.isfinite(r):
                            eqs.append(float(r))
                    pr, _ = wrapper.predict(rot)
                    flips.append(0.0 if int(pr) == int(c) else 1.0)
                    drops.append(1.0 - wrapper.score(rot, c) / max(conf0, 1e-8))
                except Exception:
                    continue
            if not eqs:
                state['i_next'] = k + 1; continue
            eq = float(np.mean(eqs))

            state['peum'].append(peum)
            state['instability'].append(1.0 - eq)
            state['eq_gradcam'].append(eq)
            state['correct'].append(1.0 if int(c) == int(true_labs[idx]) else 0.0)
            state['conf0'].append(float(conf0))
            state['conf_drop_mean'].append(float(np.mean(drops)) if drops else float('nan'))
            state['flip_rate'].append(float(np.mean(flips)) if flips else float('nan'))
            state['cam_disagree'].append(cam_disagree)
        except Exception:
            pass
        state['i_next'] = k + 1
        if (k + 1) % SAVE_EVERY == 0:
            finalize()
            n = len(state['peum'])
            if n > 20:
                r, _ = pearsonr(state['peum'], state['instability'])
                print(f'  [{k+1}/{len(spread)}] n={n} r(peum,instab)={r:.3f}  '
                      f'{(time.time()-t0)/60:.1f}m', flush=True)

    obj = finalize()
    n = obj['n_records']
    print(f'\n[DONE] {args.backbone}  n={n} records')
    if n > 20:
        for f in ('peum', 'conf_drop_mean', 'flip_rate', 'cam_disagree'):
            v = np.array(state[f], float); y = np.array(state['instability'], float)
            ok = np.isfinite(v) & np.isfinite(y)
            if ok.sum() > 20:
                r, _ = pearsonr(v[ok], y[ok])
                print(f'  r({f:15s}, instability) = {r:+.3f}')
    print(f'[SAVED] {out_path}', flush=True)
    wrapper.remove_hooks()


if __name__ == '__main__':
    main()
