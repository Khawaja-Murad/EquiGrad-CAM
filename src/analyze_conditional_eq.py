"""Equivariance conditioned on the model's own rotation stability, all backbones.

Reports three quantities on the SAME image-angle pairs, so they differ only in
conditioning and not in sample:
  Eq_all     unconditional (comparable with prior work)
  Eq_stable  restricted to rotations that leave the top-1 unchanged
  Eq_cw      continuously confidence-weighted (relaxes the requirement in
             proportion to how far the model's response moved)

All three backbones use 10 chunks at evenly spaced offsets across the
class-ordered validation set, so each spans classes drawn from the whole range
rather than a prefix. ResNet-50 was re-run on this protocol on 2026-08-31: its
earlier contiguous n=2,000 run covered only classes 0-199 and inflated both its
unconditional Eq (0.769 vs 0.700) and its prediction-stable rate (74.2% vs
61.8%), so those figures must not be reused.
"""
import glob, json, os
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_imagenet_official')
OUT = os.path.join(R, 'CONDITIONAL_EQ_SUMMARY.json')
ANG = ['15.0', '30.0', '45.0', '60.0', '90.0', '135.0', '180.0']


def collect(paths):
    g_all, e_all, g_st, e_st, gv, ev, cf = [], [], [], [], [], [], []
    n_img = 0
    for p in paths:
        d = json.load(open(p))
        bi = d.get('by_index', {})
        n_img += len(bi)
        for rec in bi.values():
            for a in ANG:
                gg, ee = rec['grad'].get(a), rec['equi'].get(a)
                st, c = rec['stable'].get(a), rec['conf'].get(a)
                if gg is not None and np.isfinite(gg):
                    g_all.append(gg)
                    if st:
                        g_st.append(gg)
                if ee is not None and np.isfinite(ee):
                    e_all.append(ee)
                    if st:
                        e_st.append(ee)
                if None not in (gg, ee, c) and all(np.isfinite([gg, ee, c])):
                    gv.append(gg); ev.append(ee); cf.append(c)
    gv, ev, cf = map(np.array, (gv, ev, cf))
    wsum = cf.sum()
    return {
        'n_images': n_img, 'n_pairs': len(g_all),
        'stable_rate': len(g_st) / max(len(g_all), 1),
        'grad': {'eq_all': float(np.mean(g_all)), 'eq_stable': float(np.mean(g_st)),
                 'eq_confweighted': float((gv * cf).sum() / wsum)},
        'equi': {'eq_all': float(np.mean(e_all)), 'eq_stable': float(np.mean(e_st)),
                 'eq_confweighted': float((ev * cf).sum() / wsum)},
    }


def main():
    out = {'note': __doc__.strip(), 'backbones': {}}
    # ResNet-50 now uses the SAME 10-chunk spread protocol as VGG/ViT. The old
    # single-file run was a contiguous n=2,000 PREFIX (classes 0-199), which
    # inflated its Eq and stable-rate; it is kept only as a fallback.
    rn50 = sorted(glob.glob(os.path.join(R, 'resnet50__R2_conditional__c*.json')))
    if not rn50:
        rn50 = [os.path.join(R, 'resnet50__R2_conditional.json')]
        print('[WARN] R50 falling back to the contiguous PREFIX run', flush=True)
    src = {'resnet50': rn50,
           'vgg16': sorted(glob.glob(os.path.join(R, 'vgg16__R2_conditional__c*.json'))),
           'vit_b_16': sorted(glob.glob(os.path.join(R, 'vit_b_16__R2_conditional__c*.json')))}
    hdr = '%-10s %6s %7s | %-24s | %-24s'
    print(hdr % ('backbone', 'n_img', 'stable', 'Grad-CAM  all/stab/cw',
                 'EquiGrad-CAM  all/stab/cw'))
    for bb, paths in src.items():
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            continue
        r = collect(paths)
        r['n_chunks'] = len(paths)
        out['backbones'][bb] = r
        g, e = r['grad'], r['equi']
        print('%-10s %6d %6.1f%% | %.3f / %.3f / %.3f       | %.3f / %.3f / %.3f'
              % (bb, r['n_images'], 100 * r['stable_rate'],
                 g['eq_all'], g['eq_stable'], g['eq_confweighted'],
                 e['eq_all'], e['eq_stable'], e['eq_confweighted']))
    tmp = OUT + '.tmp'
    json.dump(out, open(tmp, 'w'), indent=2)
    os.replace(tmp, OUT)
    print(f'\n[SAVED] {OUT}')


if __name__ == '__main__':
    main()
