#!/usr/bin/env python3
"""
exp_causal_drift.py -- is the rotation-induced saliency drift CAUSALLY faithful?
================================================================================
A causal spatial-evidence test: do the regions whose saliency changes under
rotation actually change in the model's computation? The other measurements in
the signal-vs-noise section are correlational; this one intervenes.

IDEA. Grad-CAM's map moves when the input rotates. Take the region where it moves
-- the DRIFT region -- and ask the model, causally, whether that region carries
evidence it uses DIFFERENTLY at the two orientations.

  If the drift is FAITHFUL SIGNAL, the model genuinely attends there at one
  orientation and not the other, so occluding that region should hurt the class
  score at one orientation and not the other  =>  LARGE |d0 - dtheta|.

  If the drift is OPERATOR NOISE, the region has essentially the same (and small)
  causal importance at both orientations  =>  SMALL |d0 - dtheta|, statistically
  indistinguishable from occluding a random region of equal area.

PROTOCOL. Regions are defined ONCE in the canonical frame and then transported
forward, so both occlusions cover the same image content:
  h0        = GradCAM(x, c)                        (canonical frame)
  h_theta   = GradCAM(R_theta x, c)                (rotated frame)
  h_tilde   = R_theta^-1 h_theta                   (aligned back to canonical)
  DRIFT     D = |h0 - h_tilde|      <- where the explanation moved
  AGREEMENT S = min(h0, h_tilde)    <- where both explanations agree  (control)
  RANDOM    R = random region of equal area                          (control)
For each region M (top-k% of the map): occlude x with M -> d0
                                       occlude R_theta x with R_theta M -> dtheta
Occlusion is the Gaussian-blur baseline (kernel 51, sigma=10) already used by
ca2_complete_eval.insertion_deletion, so this experiment inherits the paper's
existing missingness-bias position rather than re-opening that argument.
d = score_c(clean) - score_c(occluded), score being the softmax probability of the
FIXED original top-1 class c (the paper's target-class convention).

WHAT WOULD FALSIFY OUR CLAIM: asym(DRIFT) significantly larger than asym(RANDOM).
We report the comparison either way, with image-level bootstrap CIs and a paired
one-sided Wilcoxon test. We also report the raw magnitudes, because if the drift
region turns out to be causally unimportant at BOTH orientations that is itself
the answer: the operator is moving mass around in regions the model is not using.

SAMPLING: class-spread (i % 10 < 2), never a contiguous prefix.
Output: results_imagenet_official/<backbone>__CAUSALDRIFT.json
        atomic save every 25 images; resume via i_next.

Usage: exp_causal_drift.py --backbone resnet50 [--n_eq 1000] [--topk 0.10]
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from torchvision import transforms

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
SAVE_EVERY = 25
SIGMA      = 10
REGIONS    = ['drift', 'agree', 'random']


def save_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def topk_mask(m, frac):
    """Binary mask of the top-frac fraction of pixels of map m (H,W)."""
    H, W = m.shape
    k = max(1, int(round(frac * H * W)))
    flat = m.ravel()
    idx = np.argpartition(-flat, k - 1)[:k]
    mask = np.zeros(H * W, dtype=np.float32)
    mask[idx] = 1.0
    return mask.reshape(H, W)


def occlude(img, blurred, mask_t):
    """Replace masked pixels with the blurred baseline (mask 1 = occluded)."""
    keep = 1.0 - mask_t
    return img * keep + blurred * mask_t


def bootstrap_ci(vals, n_boot=2000, seed=0):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if v.size < 2:
        return (None, None)
    rs = np.random.RandomState(seed)
    means = [v[rs.randint(0, v.size, v.size)].mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backbone', required=True,
                    choices=('resnet50', 'vgg16', 'vit_b_16'))
    ap.add_argument('--n_eq', type=int, default=1000)
    ap.add_argument('--topk', type=float, default=0.10)
    args = ap.parse_args()

    out_path = os.path.join(OUT_DIR, f'{args.backbone}__CAUSALDRIFT.json')
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
    print(f'[SAMPLE] class-spread {len(spread)} imgs, top-{args.topk:.0%} regions', flush=True)

    wrapper = BackboneWrapper(args.backbone, device)
    labs = get_model_predictions(wrapper, imgs, labs)
    gc = _make_caller(GradCAM_v2(wrapper))
    blur = transforms.GaussianBlur(kernel_size=51, sigma=SIGMA)

    state = {r: {'d0': [], 'dth': [], 'asym': []} for r in REGIONS}
    state['i_next'] = 0
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
        for r in REGIONS:
            a = np.array(state[r]['asym'], float)
            d0 = np.array(state[r]['d0'], float)
            dt = np.array(state[r]['dth'], float)
            lo, hi = bootstrap_ci(a)
            res[r] = {
                'n_pairs': int(a.size),
                'asymmetry_mean': float(a.mean()) if a.size else None,
                'asymmetry_ci95': [lo, hi],
                'drop_at_0deg_mean': float(d0.mean()) if d0.size else None,
                'drop_at_theta_mean': float(dt.mean()) if dt.size else None,
            }
        # paired one-sided Wilcoxon: is DRIFT asymmetry > RANDOM asymmetry?
        tests = {}
        try:
            from scipy.stats import wilcoxon
            for r in ('drift', 'agree'):
                x = np.array(state[r]['asym'], float)
                y = np.array(state['random']['asym'], float)
                n = min(x.size, y.size)
                if n > 10:
                    st, p = wilcoxon(x[:n], y[:n], alternative='greater')
                    tests[f'{r}_gt_random'] = {'p': float(p), 'n': int(n),
                                               'median_diff': float(np.median(x[:n] - y[:n]))}
        except Exception as e:
            tests['error'] = str(e)
        obj = {
            'config': {'backbone': args.backbone, 'n_eq': len(spread),
                       'topk_frac': args.topk, 'blur_sigma': SIGMA,
                       'eq_angles': EQ_ANGLES, 'seed': 42,
                       'sampling': f'class-spread {PER_CLASS}/class',
                       'score': 'softmax prob of fixed original top-1 class',
                       'reading': 'faithful drift => asym(drift) >> asym(random); '
                                  'operator noise => asym(drift) ~ asym(random)'},
            'result': res, 'tests': tests, 'state': state,
        }
        save_atomic(out_path, obj)
        return obj

    rs = np.random.RandomState(42)
    t0 = time.time()
    for k in range(state['i_next'], len(spread)):
        idx = spread[k]
        img, c = imgs[idx], labs[idx]
        try:
            h0 = gc(img, c)
            if h0.std() < 1e-8:
                state['i_next'] = k + 1; continue
            blurred0 = blur(img.unsqueeze(0)).squeeze(0)
            s0_clean = wrapper.score(img, c)
        except Exception:
            state['i_next'] = k + 1; continue

        for a in EQ_ANGLES:
            ang = float(a)
            try:
                rot = rpbh_rotate(img, ang)
                hth = gc(rot, c)
                if hth.std() < 1e-8:
                    continue
                # align the rotated-input map back to canonical
                htil = inv_rotate(torch.from_numpy(hth).unsqueeze(0), ang).squeeze(0).numpy()
                D = np.abs(h0 - htil)
                S = np.minimum(h0, htil)
                masks = {
                    'drift':  topk_mask(D, args.topk),
                    'agree':  topk_mask(S, args.topk),
                    'random': topk_mask(rs.rand(*h0.shape).astype(np.float32), args.topk),
                }
                blurredt = blur(rot.unsqueeze(0)).squeeze(0)
                st_clean = wrapper.score(rot, c)

                for rname, M in masks.items():
                    Mt = torch.from_numpy(M).unsqueeze(0)
                    # canonical-frame occlusion
                    d0 = s0_clean - wrapper.score(occlude(img, blurred0, Mt), c)
                    # same region transported into the rotated frame
                    Mrot = inv_rotate(Mt, -ang).clamp(0, 1)
                    dth = st_clean - wrapper.score(occlude(rot, blurredt, Mrot), c)
                    state[rname]['d0'].append(float(d0))
                    state[rname]['dth'].append(float(dth))
                    state[rname]['asym'].append(float(abs(d0 - dth)))
            except Exception:
                continue

        state['i_next'] = k + 1
        if (k + 1) % SAVE_EVERY == 0:
            finalize()
            def m(r):
                v = state[r]['asym']
                return np.mean(v) if v else float('nan')
            print(f'  [{k+1}/{len(spread)}] asym drift={m("drift"):.4f} '
                  f'agree={m("agree"):.4f} random={m("random"):.4f}  '
                  f'{(time.time()-t0)/60:.1f}m', flush=True)

    obj = finalize()
    print('\n' + '=' * 78)
    print(f'CAUSAL DRIFT TEST  {args.backbone}  (n={len(spread)}, top-{args.topk:.0%})')
    for r in REGIONS:
        v = obj['result'][r]
        ci = v['asymmetry_ci95']
        print(f'  {r:7s} asym={v["asymmetry_mean"]:.4f} '
              f'CI95=[{ci[0]:.4f},{ci[1]:.4f}]  '
              f'drop@0={v["drop_at_0deg_mean"]:.4f} drop@th={v["drop_at_theta_mean"]:.4f}  '
              f'n={v["n_pairs"]}')
    print(f'  tests: {json.dumps(obj["tests"])}')
    print(f'[SAVED] {out_path}', flush=True)
    wrapper.remove_hooks()


if __name__ == '__main__':
    main()
