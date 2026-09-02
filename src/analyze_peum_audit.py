#!/usr/bin/env python
"""
analyze_peum_audit.py - downstream statistics for exp_peum_audit.py.

CPU-only, no GPU and no model: reads the raw per-image records written by
exp_peum_audit.py and computes the hardened PEUM claims.

Addresses the three standard objections to a triage claim:
  1. CIRCULARITY  - the operating point (top-decile threshold) is chosen on a
     held-out half and the lift is reported on the other half.
  2. NO UNCERTAINTY - every headline number carries an image-level bootstrap CI.
  3. "ONLY AN INSTABILITY DETECTOR" - PEUM is scored against actual
     misclassification (AUROC), not just against explanation instability, and
     against three baselines that are cheaper than it.

Usage:
  python analyze_peum_audit.py --backbone resnet50 [--n_boot 10000] [--seed 0]
"""
import argparse, json, os
import numpy as np

RES_DIR = os.path.expanduser('~/scratch/ca2gradcam/results_imagenet_official')
SIGNALS = ['peum', 'cam_disagree', 'conf_drop_mean', 'flip_rate']


# ---------------------------------------------------------------- statistics
def spearman(x, y):
    """Spearman rho via Pearson on ranks (average ranks for ties)."""
    return pearson(rankdata(x), rankdata(y))


def pearson(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float('nan')


def rankdata(a):
    """Average ranks, ties shared - matches scipy.stats.rankdata('average')."""
    a = np.asarray(a, float)
    order = a.argsort(kind='mergesort')
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average over tie groups
    s = a[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def auroc(score, positive):
    """P(score[pos] > score[neg]), ties counted as 1/2 (Mann-Whitney U)."""
    pos = positive.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    r = rankdata(score)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def decile_lift(signal, target, thresh):
    """Mean target above `thresh`, divided by the mean over all images."""
    sel = signal >= thresh
    if sel.sum() == 0 or target.mean() == 0:
        return float('nan')
    return float(target[sel].mean() / target.mean())


def boot_ci(fn, n, rng, n_boot, alpha=0.05):
    """Image-level bootstrap: resample IMAGES, recompute, percentile CI."""
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(n_boot)])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (float('nan'), float('nan'))
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='resnet50')
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--top_frac', type=float, default=0.10)
    args = ap.parse_args()

    path = os.path.join(RES_DIR, f'{args.backbone}__PEUM_AUDIT.json')
    d = json.load(open(path))
    per = d['per_image']

    cols = SIGNALS + ['instability', 'correct']
    A = {c: np.asarray(per[c], float) for c in cols}
    ok = np.ones(len(A['peum']), bool)
    for c in cols:
        ok &= np.isfinite(A[c])
    A = {c: v[ok] for c, v in A.items()}
    n = int(ok.sum())

    inst = A['instability']
    wrong = 1.0 - A['correct']          # 1 = model got it wrong
    rng = np.random.default_rng(args.seed)

    print(f'=== PEUM audit: {args.backbone} ===')
    print(f'records {len(ok)} -> usable {n} (dropped {int((~ok).sum())} non-finite)')
    print(f'mean instability {inst.mean():.4f} | misclassification rate {wrong.mean():.4f}')
    print(f'bootstrap {args.n_boot} resamples, top-{args.top_frac:.0%} operating point\n')

    # ---- 1. association with instability, whole sample, with CIs -----------
    print('--- 1. association with explanation instability (1 - Eq) ---')
    print(f"{'signal':<16}{'pearson r':>12}{'95% CI':>20}{'spearman':>12}")
    rows = {}
    for s in SIGNALS:
        x = A[s]
        r = pearson(x, inst)
        lo, hi = boot_ci(lambda idx: pearson(x[idx], inst[idx]), n, rng, args.n_boot)
        rows[s] = (r, lo, hi, spearman(x, inst))
        print(f'{s:<16}{r:>12.3f}{f"[{lo:.3f}, {hi:.3f}]":>20}{rows[s][3]:>12.3f}')

    # ---- 2. held-out top-decile lift --------------------------------------
    # Threshold is FIT on half A and APPLIED to half B, so the operating point
    # is never chosen on the images it is scored on.
    print(f'\n--- 2. held-out top-{args.top_frac:.0%} lift on instability ---')
    print('    (threshold fit on split A, lift measured on split B)')
    perm = rng.permutation(n)
    a_idx, b_idx = perm[:n // 2], perm[n // 2:]
    print(f"{'signal':<16}{'lift (held-out)':>18}{'95% CI':>20}{'in-sample':>12}")
    for s in SIGNALS:
        x = A[s]
        thr = np.quantile(x[a_idx], 1 - args.top_frac)
        lift = decile_lift(x[b_idx], inst[b_idx], thr)
        insample = decile_lift(x, inst, np.quantile(x, 1 - args.top_frac))

        def f(idx, x=x, thr=thr):
            bb = b_idx[idx % len(b_idx)]
            return decile_lift(x[bb], inst[bb], thr)

        lo, hi = boot_ci(f, len(b_idx), rng, args.n_boot)
        print(f'{s:<16}{lift:>18.3f}{f"[{lo:.3f}, {hi:.3f}]":>20}{insample:>12.3f}')

    # ---- 3. does it predict actual misclassification? ---------------------
    print('\n--- 3. AUROC for predicting MISCLASSIFICATION (not instability) ---')
    print('    0.5 = chance. This is the "audit signal vs instability detector" test.')
    print(f"{'signal':<16}{'AUROC':>10}{'95% CI':>20}")
    for s in SIGNALS:
        x = A[s]
        au = auroc(x, wrong)
        lo, hi = boot_ci(lambda idx: auroc(x[idx], wrong[idx]), n, rng, args.n_boot)
        print(f'{s:<16}{au:>10.3f}{f"[{lo:.3f}, {hi:.3f}]":>20}')

    au_i = auroc(inst, wrong)
    lo, hi = boot_ci(lambda idx: auroc(inst[idx], wrong[idx]), n, rng, args.n_boot)
    print(f'{"(instability)":<16}{au_i:>10.3f}{f"[{lo:.3f}, {hi:.3f}]":>20}   reference')

    # ---- 4. paired: does PEUM beat each baseline? -------------------------
    # Bootstrap the DIFFERENCE on the same resamples, so the comparison is paired.
    print('\n--- 4. PEUM minus baseline, paired bootstrap (CI excluding 0 = wins) ---')
    print(f"{'comparison':<28}{'d(r_inst)':>12}{'95% CI':>20}{'d(AUROC)':>11}{'95% CI':>20}")
    p = A['peum']
    for s in SIGNALS[1:]:
        x = A[s]
        dr = pearson(p, inst) - pearson(x, inst)
        lo1, hi1 = boot_ci(
            lambda idx: pearson(p[idx], inst[idx]) - pearson(x[idx], inst[idx]),
            n, rng, args.n_boot)
        da = auroc(p, wrong) - auroc(x, wrong)
        lo2, hi2 = boot_ci(
            lambda idx: auroc(p[idx], wrong[idx]) - auroc(x[idx], wrong[idx]),
            n, rng, args.n_boot)
        print(f'{"peum - " + s:<28}{dr:>12.3f}{f"[{lo1:.3f}, {hi1:.3f}]":>20}'
              f'{da:>11.3f}{f"[{lo2:.3f}, {hi2:.3f}]":>20}')


if __name__ == '__main__':
    main()
