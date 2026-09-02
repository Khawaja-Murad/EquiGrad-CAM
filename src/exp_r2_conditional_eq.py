#!/usr/bin/env python3
"""
exp_r2_conditional_eq.py -- Prediction-stratified rotational equivariance (ResNet-50)
=====================================================================================
Question: is Grad-CAM's rotational drift merely an argmax artefact (the model's
top-1 flips under rotation, so of course the saliency map moves), or does the
drift PERSIST even on the subset of rotations where the prediction is unchanged?

DATASET  : full official ILSVRC-2012 ImageNet-1K validation set, deterministic
           seed=42 slice (load_imagenet_val, n_per_class=10 x 1000 classes =
           10,000 imgs; Resize(256)+CenterCrop(224)). IDENTICAL ordering to
           run_official_one.py, so per-image results align with the baselines.
METRIC   : per-(image,angle) equivariance
             Eq(i,a) = Pearson( M(R_a . x_i) ,  R_a . M(x_i) )
           over EQ_ANGLES = [15,30,45,60,90,135,180], for M in
             {Grad-CAM (GradCAM_v2), EquiGrad-CAM tau=0 (iGradCAM, 18 views)}.
           Degenerate (sigma < 1e-8) maps -> Eq = None (mirrors v11 eq_scores /
           per_image_eq_one degeneracy handling). Each (i,a) pair is stratified by
           prediction stability
             stable(i,a) = ( argmax f(R_a . x_i) == c_i ),  c_i = model top-1 on x_i.
           Reported per method: mean Eq in the STABLE vs FLIPPED stratum, stratum
           sizes, and Pearson(1-Eq, 1-conf) WITHIN the stable stratum. conf is the
           top-1 softmax probability from wrapper.predict; inside the stable stratum
           the top-1 IS c_i, so conf == prob(c_i) there (forward-only, no backward).
RUNTIME  : ~8 h for the full 10,000 imgs on one A100. Cost is dominated by
           EquiGrad tau=0: 8 aggregations x 18 fwd+bwd per image ~= 2.8 s/img
           (measured from the egtau0c ResNet-50 chunk logs), plus Grad-CAM
           (~0.15 s/img) and 7 forward-only predicts (~0.06 s/img). RESUMABLE
           (i_next cursor, atomic tmp+os.replace save every 100 imgs): a 2 h slice
           does ~2.4k imgs, just resubmit. Or parallelise with --start/--count
           (~50 min per 1k chunk, 10 chunks); chunk JSONs are keyed by ABSOLUTE
           image index and merge cleanly (never join by list position).

PRE-REGISTERED CLAIM (do NOT overclaim it to "all drift is noise"):
  "Drift PERSISTS in the prediction-stable stratum -- Grad-CAM's mean Eq there
   stays well below EquiGrad-CAM(tau=0)'s -- so the instability is NOT an argmax
   artefact but is intrinsic to the single-view pipeline."

CONFIRMS the contribution: the stable-stratum gap (EquiGrad - Grad-CAM) stays
  large and positive => single-view Grad-CAM is rotation-unstable even when the
  decision is unchanged, so multi-view feature-space aggregation is warranted.
THREATENS the contribution: if that gap collapsed toward 0 (Grad-CAM's Eq rising
  to EquiGrad's once prediction flips are excluded), the drift would be an argmax
  artefact and a trivial "reject prediction-flipping rotations" filter would
  suffice, undercutting the aggregation contribution.

FROZEN v11 (ca2_complete_eval.py) is imported, NEVER edited.
Output: results_imagenet_official/<bb>__R2_conditional.json
        (chunk mode -> ...__R2_conditional__c<start>.json)
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, os.path.expanduser('~/scratch/ca2gradcam'))
from ca2_complete_eval import (
    set_seed, BackboneWrapper,
    GradCAM_v2, iGradCAM,
    load_imagenet_val, get_model_predictions,
    rpbh_rotate, rotate_heatmap_np, _make_caller,
)
from run_official_one import VAL_DIR, EQ_ANGLES  # identical data path + angle set

OUT_DIR = os.path.expanduser('~/scratch/ca2gradcam/results_imagenet_official')
DEGEN = 1e-8


def _f3(v):
    return 'NA' if v is None else f'{v:.3f}'


def per_angle_eq(hm_r, ref):
    """Pearson(heatmap(rot), rotate(heatmap0)); None on degenerate maps (v11 rule)."""
    if hm_r.std() < DEGEN or ref.std() < DEGEN:
        return None
    try:
        p, _ = pearsonr(hm_r.flatten(), ref.flatten())
    except Exception:
        return None
    return float(p) if np.isfinite(p) else None


def _mean_or_none(xs):
    return float(np.mean(xs)) if xs else None


def compute_summary(by_index, angles):
    """Stratify every finite (image,angle) pair by prediction stability and report
    Grad-CAM / EquiGrad mean Eq per stratum + Pearson(1-Eq,1-conf) in the stable
    stratum. Everything is keyed off the per-image records already keyed by
    absolute index -- no list-position joins."""
    akeys = [str(a) for a in angles]
    rows = []  # (grad_eq|None, equi_eq|None, stable_bool, conf_float)
    for rec in by_index.values():
        for a in akeys:
            st = rec['stable'].get(a)
            cf = rec['conf'].get(a)
            if st is None or cf is None:
                continue
            rows.append((rec['grad'].get(a), rec['equi'].get(a), bool(st), float(cf)))

    def method_block(sel):  # sel: 0 -> grad, 1 -> equi
        s_eq, s_inst, s_unc, f_eq = [], [], [], []
        for g, e, st, cf in rows:
            v = g if sel == 0 else e
            if v is None:
                continue
            if st:
                s_eq.append(v); s_inst.append(1.0 - v); s_unc.append(1.0 - cf)
            else:
                f_eq.append(v)
        blk = {
            'stable':  {'mean_eq': _mean_or_none(s_eq), 'n': len(s_eq)},
            'flipped': {'mean_eq': _mean_or_none(f_eq), 'n': len(f_eq)},
        }
        if len(s_inst) >= 2 and np.std(s_inst) > 0 and np.std(s_unc) > 0:
            r, p = pearsonr(s_inst, s_unc)
            blk['pearson_1eq_1conf_stable'] = {'r': float(r), 'p': float(p), 'n': len(s_inst)}
        else:
            blk['pearson_1eq_1conf_stable'] = {'r': None, 'p': None, 'n': len(s_inst)}
        return blk

    grad_blk = method_block(0)
    equi_blk = method_block(1)

    # per-angle stability + stable-stratum means (paper-friendly breakdown)
    per_angle = {}
    for a in akeys:
        gs, es, ns, nt = [], [], 0, 0
        for rec in by_index.values():
            st = rec['stable'].get(a)
            if st is None:
                continue
            nt += 1
            if st:
                ns += 1
                if rec['grad'].get(a) is not None:
                    gs.append(rec['grad'][a])
                if rec['equi'].get(a) is not None:
                    es.append(rec['equi'][a])
        per_angle[a] = {
            'stable_rate': (ns / nt) if nt else None,
            'grad_stable_mean_eq': _mean_or_none(gs),
            'equi_stable_mean_eq': _mean_or_none(es),
            'n': nt,
        }

    n_pairs = len(rows)
    n_stable = sum(1 for _, _, st, _ in rows if st)
    gsm = grad_blk['stable']['mean_eq']
    esm = equi_blk['stable']['mean_eq']
    return {
        'n_images': len(by_index),
        'n_pairs': n_pairs,
        'n_stable_pairs': n_stable,
        'n_flipped_pairs': n_pairs - n_stable,
        'stable_rate': (n_stable / n_pairs) if n_pairs else None,
        'grad': grad_blk,
        'equi': equi_blk,
        'stable_stratum_gap_equi_minus_grad':
            (esm - gsm) if (gsm is not None and esm is not None) else None,
        'per_angle': per_angle,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backbone', default='resnet50')
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--count', type=int, default=None)
    ap.add_argument('--n_angles', type=int, default=18)  # EquiGrad tau=0 headline
    args = ap.parse_args()

    bb = args.backbone
    n = args.n
    start = args.start
    is_chunk = (args.count is not None) or (start != 0)
    end = min(start + args.count, n) if args.count is not None else n

    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f'__c{start}' if is_chunk else ''
    out_path = os.path.join(OUT_DIR, f'{bb}__R2_conditional{suffix}.json')

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    imgs, labs = imgs[:n], labs[:n]
    end = min(end, len(imgs))
    print(f'[DATA] {len(imgs)} official ImageNet val imgs; '
          f'processing absolute idx [{start}:{end})', flush=True)

    wrapper = BackboneWrapper(bb, device)
    # Model top-1 c_i on the UNROTATED image. Computed only for the chunk range;
    # get_model_predictions is per-image deterministic, so c_i for absolute index
    # i == start + j is identical whether or not the full set is passed.
    sub_labs = get_model_predictions(wrapper, imgs[start:end], labs[start:end])

    gc_call = _make_caller(GradCAM_v2(wrapper))
    eq_call = _make_caller(iGradCAM(wrapper, n_angles=args.n_angles))

    state = {
        'by_index': {},
        'i_next': start,
        'config': {
            'backbone': bb, 'val_dir': VAL_DIR, 'n': n, 'start': start, 'end': end,
            'eq_angles': EQ_ANGLES, 'n_angles': args.n_angles, 'seed': 42,
            'metric': 'per-(image,angle) Pearson equivariance, stratified by '
                      'prediction stability; conf=top-1 prob (=prob(c) in stable stratum)',
        },
    }
    if os.path.exists(out_path):
        try:
            loaded = json.load(open(out_path))
            state['by_index'] = loaded.get('by_index', {})
            state['i_next'] = int(loaded.get('i_next', start))
            print(f'[RESUME] {out_path}: {len(state["by_index"])} imgs done, '
                  f'i_next={state["i_next"]}', flush=True)
        except Exception as e:
            print(f'[RESUME] failed ({e}); starting fresh', flush=True)

    def save():
        state['summary'] = compute_summary(state['by_index'], EQ_ANGLES)
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, out_path)

    t0 = time.time()
    for j in range(end - start):
        i = start + j
        if i < state['i_next']:
            continue
        img = imgs[i]
        c = int(sub_labs[j])
        try:
            grad_hm0 = gc_call(img, c)
            equi_hm0 = eq_call(img, c)
            grad_ok = grad_hm0.std() >= DEGEN
            equi_ok = equi_hm0.std() >= DEGEN
            rec = {'grad': {}, 'equi': {}, 'stable': {}, 'conf': {}}
            for a in EQ_ANGLES:
                ak = str(a)
                rot = rpbh_rotate(img, float(a))       # pad=64 default, matches v11
                pred_a, conf_a = wrapper.predict(rot)  # forward-only (no backward)
                rec['stable'][ak] = bool(pred_a == c)
                rec['conf'][ak] = float(conf_a)
                if grad_ok:
                    rec['grad'][ak] = per_angle_eq(
                        gc_call(rot, c), rotate_heatmap_np(grad_hm0, float(a)))
                else:
                    rec['grad'][ak] = None
                if equi_ok:
                    rec['equi'][ak] = per_angle_eq(
                        eq_call(rot, c), rotate_heatmap_np(equi_hm0, float(a)))
                else:
                    rec['equi'][ak] = None
            state['by_index'][str(i)] = rec
        except Exception as e:
            print(f'  [IMG {i}] error: {e}', flush=True)
        state['i_next'] = i + 1
        if (i + 1) % 100 == 0:
            save()
            s = state['summary']
            print(f'  [{i+1}/{end}] stable_rate={_f3(s["stable_rate"])}  '
                  f'grad_stable_eq={_f3(s["grad"]["stable"]["mean_eq"])}  '
                  f'equi_stable_eq={_f3(s["equi"]["stable"]["mean_eq"])}  '
                  f'{(time.time()-t0)/60:.1f}m', flush=True)

    save()
    s = state['summary']
    g, e = s['grad'], s['equi']
    print(f'\n[SAVED] {out_path}', flush=True)
    print('=' * 72, flush=True)
    print(f'  images={s["n_images"]}  pairs={s["n_pairs"]}  '
          f'stable_rate={_f3(s["stable_rate"])} '
          f'({s["n_stable_pairs"]} stable / {s["n_flipped_pairs"]} flipped)', flush=True)
    print(f'  Grad-CAM      stable Eq={_f3(g["stable"]["mean_eq"])} (n={g["stable"]["n"]})   '
          f'flipped Eq={_f3(g["flipped"]["mean_eq"])} (n={g["flipped"]["n"]})', flush=True)
    print(f'  EquiGrad tau0 stable Eq={_f3(e["stable"]["mean_eq"])} (n={e["stable"]["n"]})   '
          f'flipped Eq={_f3(e["flipped"]["mean_eq"])} (n={e["flipped"]["n"]})', flush=True)
    print(f'  stable-stratum gap (EquiGrad - Grad-CAM) = '
          f'{_f3(s["stable_stratum_gap_equi_minus_grad"])}', flush=True)
    print(f'  Grad-CAM  Pearson(1-Eq,1-conf | stable) = {g["pearson_1eq_1conf_stable"]}', flush=True)
    print(f'  EquiGrad  Pearson(1-Eq,1-conf | stable) = {e["pearson_1eq_1conf_stable"]}', flush=True)
    gap = s['stable_stratum_gap_equi_minus_grad']
    if gap is not None:
        if gap > 0.05:
            print('  -> PRE-REGISTERED CLAIM SUPPORTED: drift PERSISTS in the stable '
                  'stratum; Grad-CAM instability is not an argmax artefact.', flush=True)
        else:
            print('  -> gap small; pre-registered claim NOT clearly supported at this n '
                  '(would threaten the aggregation contribution -- inspect before trusting).',
                  flush=True)

    wrapper.remove_hooks()


if __name__ == '__main__':
    main()
