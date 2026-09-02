#!/usr/bin/env python3
"""
run_rise_vit.py -- RISE on ViT-B/16. Fixes a factual error in the paper.
========================================================================
Table 2 currently tells the reader that Score-CAM / RISE / Aug. Score-CAM are
"patch-token-incompatible" on ViT and prints a dash. That justification is FALSE
for RISE: RISE is a black-box method that estimates importance by probing the
model with randomly masked INPUTS and reading the output score. It never touches
feature maps or patch tokens, so it applies to a ViT exactly as to a CNN. The
real reason it was not run was compute budget, and the paper must say so.

This matters because ViT-B/16 is where our largest relative gain is claimed
(0.250 -> 0.867) and where we currently field no model-agnostic baseline at all.

SCOPE REDUCTION (stated, and to be disclosed in the paper): n_eq=500 class-spread,
Eq only, no insertion/deletion. Measured rate from the existing ResNet-50 RISE run
is ~33.9 s/image for the 8 RISE maps an Eq row needs (0 deg + 7 angles, 8,000
masks each = 64k forwards/image); ViT is slower per forward, so 500 images is what
fits one 12 h job with margin. Mask count, grid and p are UNCHANGED from
rise_baseline.py so the ViT row is directly comparable to the published
ResNet-50/VGG-16 RISE rows.

SAMPLING: class-spread (i % 10 < 2), never a contiguous prefix -- the paper
documents that a prefix inflates Eq by +0.07..+0.09 on this class-blocked loader.

Reuses the RISE implementation and per-image Eq helper from rise_baseline.py
VERBATIM (imported, not copied), so this is the same estimator as the CNN rows.

Output: results_imagenet_official/vit_b_16__RISE.json
        atomic save every 10 images; resume via i_eq_next.

Usage: run_rise_vit.py [--n_eq 500]
"""
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser('~/scratch/ca2gradcam'))
from ca2_complete_eval import (set_seed, BackboneWrapper,
                               load_imagenet_val, get_model_predictions)
from rise_baseline import RISE, per_image_eq, N_MASKS, MASK_GRID, MASK_P, BATCH

VAL_DIR    = os.path.expanduser('~/scratch/ca2gradcam/imagenet_val_official')
OUT        = os.path.expanduser('~/scratch/ca2gradcam/results_imagenet_official/vit_b_16__RISE.json')
EQ_ANGLES  = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]
PER_CLASS  = 2
SAVE_EVERY = 10


def save_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n_eq', type=int, default=500)
    args = ap.parse_args()
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

    wrapper = BackboneWrapper('vit_b_16', device)
    labs = get_model_predictions(wrapper, imgs, labs)
    rise = RISE(wrapper, n_masks=N_MASKS, batch=BATCH, device=device)
    print(f'[RISE-ViT] {N_MASKS} masks ({MASK_GRID}x{MASK_GRID}, p={MASK_P}), '
          f'batch={BATCH}  -- black-box, no patch-token access', flush=True)

    def call(img, cidx):
        return rise(img, cidx)

    state = {'per_image_eq': [], 'i_eq_next': 0,
             'n_degenerate': 0, 'n_unscored': 0}
    if os.path.exists(OUT):
        try:
            old = json.load(open(OUT))
            if 'state' in old:
                state = old['state']
                print(f'[RESUME] from image {state["i_eq_next"]}', flush=True)
        except Exception as e:
            print(f'[RESUME] failed: {e}', flush=True)

    def finalize():
        arr = np.array(state['per_image_eq'], float)
        nd = state['n_degenerate']       # constant base map -> already scored 0.0
        nu = state['n_unscored']         # no angle scored at all
        obj = {
            'config': {'backbone': 'vit_b_16', 'method': 'RISE',
                       'n_masks': N_MASKS, 'grid': MASK_GRID, 'p': MASK_P,
                       'n_eq': len(spread), 'eq_angles': EQ_ANGLES, 'seed': 42,
                       'sampling': f'class-spread {PER_CLASS}/class',
                       'eq_only': True,
                       'note': 'RISE is black-box input masking; it needs no patch-token '
                               'access. Prior omission on ViT was compute budget, not '
                               'incompatibility.'},
            'result': {
                'n_scored': int(arr.size), 'n_degenerate': int(nd),
                'n_unscored': int(nu),
                # The upstream helper already assigns 0.0 to a constant base map,
                # i.e. the paper's uniform zero-fill convention, so eq is the
                # zero-filled mean. eq_excl_degenerate is the conditional mean.
                'eq': float(arr.mean()) if arr.size else None,
                'eq_excl_degenerate': (float(arr[arr > 0].mean())
                                       if (arr > 0).any() else None),
                'eq_std': float(arr.std()) if arr.size else None,
            },
            'state': state,
        }
        save_atomic(OUT, obj)
        return obj

    t0 = time.time()
    for k in range(state['i_eq_next'], len(spread)):
        idx = spread[k]
        # rise_baseline.per_image_eq returns a SCALAR: 0.0 for a degenerate base
        # map (the zero-fill convention already applied) or None if no angle
        # scored. It does not return per-angle values.
        res = per_image_eq(call, imgs[idx], labs[idx], EQ_ANGLES)
        if res is None:
            state['n_unscored'] += 1
        else:
            if float(res) == 0.0:
                state['n_degenerate'] += 1
            state['per_image_eq'].append(float(res))
        state['i_eq_next'] = k + 1
        if (k + 1) % SAVE_EVERY == 0:
            finalize()
            cur = np.mean(state['per_image_eq']) if state['per_image_eq'] else float('nan')
            el = (time.time() - t0) / 60
            done = k + 1 - 0
            print(f'  [EQ {k+1}/{len(spread)}] Eq={cur:.4f} degen={state["n_degenerate"]} '
                  f'{el:.1f}m ({el*60/max(done,1):.1f}s/img)', flush=True)

    obj = finalize()
    r = obj['result']
    print(f'\n[DONE] RISE ViT-B/16  Eq={r["eq"]:.4f} '
          f'(excl. degenerate {r["eq_excl_degenerate"]})  '
          f'n={r["n_scored"]}  degenerate={r["n_degenerate"]}  unscored={r["n_unscored"]}')
    print(f'[SAVED] {OUT}', flush=True)
    wrapper.remove_hooks()


if __name__ == '__main__':
    main()
