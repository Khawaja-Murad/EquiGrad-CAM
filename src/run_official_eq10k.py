"""Full-scale re-evaluation on the official ILSVRC/imagenet-1k val set.

Runs ins/del AND equivariance on the full 10,000-image subset (10/class × 1000
classes), with equivariance computed at 7 angles. Per-method incremental save;
resumable on rerun.

Methods covered (split by cost):
  fast group  : GradCAM, GradCAM++, XGrad-CAM, LayerCAM, AugCAM_t0, AugCAM_t01, CA2_N6, CA2_N18
  slow group  : SmoothGC++, ScoreCAM  -- handled in separate per-method jobs

Invocation:
  python run_official_eq10k.py <backbone> <group>
  backbone ∈ {resnet50, vgg16, vit_b_16}
  group    ∈ {fast, slow}

Output: results_imagenet_official/ImageNet1K_<backbone>_<group>.json
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper,
    GradCAM_v2, GradCAMPP, XGradCAM, LayerCAM, ScoreCAM,
    SmoothGradCAMPP, AugCAM_v2, CA2GradCAM_v2,
    load_imagenet_val, get_model_predictions,
    insertion_deletion, eq_scores, compute_per_image_eq, _make_caller,
)

VAL_DIR   = './imagenet_val_official'
N_IMGS    = 10000
N_EQ      = 10000
EQ_ANGLES = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]


def main():
    if len(sys.argv) < 3:
        print('usage: run_official_eq10k.py <backbone> <group>')
        sys.exit(2)
    bb, group = sys.argv[1], sys.argv[2]
    assert bb in ('resnet50', 'vgg16', 'vit_b_16')
    assert group in ('fast', 'slow')
    if bb == 'vit_b_16' and group == 'slow':
        # Slow group on ViT = SmoothGC++ only (ScoreCAM excluded for ViT)
        pass

    out_dir = './results_imagenet_official'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'ImageNet1K_{bb}_{group}.json')

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    print(f'[DATA] {len(imgs)} ImageNet val (OFFICIAL) images', flush=True)
    assert len(imgs) >= N_IMGS, f'need {N_IMGS}, got {len(imgs)}'
    imgs = imgs[:N_IMGS]
    labs = labs[:N_IMGS]

    wrapper = BackboneWrapper(bb, device)
    labs = get_model_predictions(wrapper, imgs, labs)

    if group == 'fast':
        methods = {
            'GradCAM':    GradCAM_v2(wrapper),
            'GradCAM++':  GradCAMPP(wrapper),
            'XGrad-CAM':  XGradCAM(wrapper),
            'LayerCAM':   LayerCAM(wrapper),
            'AugCAM_t0':  AugCAM_v2(wrapper, tau=0.0),
            'AugCAM_t01': AugCAM_v2(wrapper, tau=0.10),
            'CA2_N6':     CA2GradCAM_v2(wrapper, tau=0.10, n_angles=6),
            'CA2_N18':    CA2GradCAM_v2(wrapper, tau=0.10, n_angles=18),
        }
    else:  # slow
        methods = {'SmoothGC++': SmoothGradCAMPP(wrapper, n_samples=50)}
        if bb != 'vit_b_16':
            methods['ScoreCAM'] = ScoreCAM(wrapper, batch_size=64)

    # Resume support
    results = {}
    per_image_eq = {}
    if os.path.exists(out_path):
        try:
            loaded = json.load(open(out_path))
            per_image_eq = loaded.pop('_per_image_eq', {})
            results = loaded
            done = [k for k in results if isinstance(results.get(k), dict) and 'eq' in results.get(k, {})]
            print(f'[RESUME] {out_path}; skipping done: {done}', flush=True)
        except Exception as e:
            print(f'[RESUME] failed: {e}', flush=True)
            results, per_image_eq = {}, {}

    def save():
        tmp = out_path + '.tmp'
        data = {**results, '_per_image_eq': per_image_eq, 'config': {
            'backbone': bb, 'group': group, 'val_dir': VAL_DIR,
            'n': N_IMGS, 'n_eq': N_EQ, 'eq_angles': EQ_ANGLES, 'seed': 42,
        }}
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, out_path)

    for name, m in methods.items():
        if name in results and isinstance(results.get(name), dict) and 'eq' in results[name]:
            print(f'  [{name}] SKIP', flush=True)
            continue
        call = _make_caller(m)
        print(f'\n  [{name}] {N_IMGS} ins/del + {N_EQ} eq images', flush=True)
        ins_l, del_l = [], []
        t0 = time.time()
        for i, (img, cidx) in enumerate(zip(imgs, labs)):
            try:
                hm = call(img, cidx)
            except Exception as e:
                print(f'    image {i} error: {e}', flush=True)
                continue
            ii, dd = insertion_deletion(wrapper, img, hm, cidx, 20, 10)
            ins_l.append(ii); del_l.append(dd)
            if (i+1) % 500 == 0:
                print(f'    {i+1}/{N_IMGS} (Ins={np.mean(ins_l):.3f})', flush=True)
        print(f'    Equivariance ({N_EQ} imgs)...', flush=True)
        eq_dict = eq_scores(call, imgs[:N_EQ], labs[:N_EQ], EQ_ANGLES)
        per_image_eq[name] = compute_per_image_eq(call, imgs[:N_EQ], labs[:N_EQ], EQ_ANGLES)
        elapsed = time.time() - t0
        results[name] = {
            'n':         N_IMGS,
            'n_eq':      N_EQ,
            'eq':        float(eq_dict['pearson']),
            'eq_std':    float(eq_dict['pearson_std']),
            'ins':       float(np.mean(ins_l)),
            'ins_std':   float(np.std(ins_l)),
            'del':       float(np.mean(del_l)),
            'del_std':   float(np.std(del_l)),
            'time_img':  elapsed / max(1, len(ins_l)),
            'per_angle': eq_dict['per_angle'],
        }
        save()
        r = results[name]
        print(f'  [{name}] Eq={r["eq"]:.3f}±{r["eq_std"]:.3f}  Ins={r["ins"]:.3f}  Del={r["del"]:.3f}  '
              f'({elapsed/3600:.2f}h, t/img={r["time_img"]:.2f}s)', flush=True)

    print(f'\n[SAVED] {out_path}', flush=True)


if __name__ == '__main__':
    main()
