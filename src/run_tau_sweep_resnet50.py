"""PCF τ sweep on ResNet-50 / ImageNet-1K val (Table 8 of the paper).

Six τ values × CA² N=18 × 2000 ins/del images × 500 eq images × 7 angles.
Records Eq, Ins, Del with std for each τ.

Output: ./results_imagenet/TAU_SWEEP_resnet50.json
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper, CA2GradCAM_v2,
    load_imagenet_val, get_model_predictions,
    insertion_deletion, eq_scores, _make_caller,
)

TAU_VALUES = [0.0, 0.05, 0.10, 0.20, 0.50, 0.80]
N_INS_DEL  = 2000   # paper used 1K Tiny ImageNet; we sample 2K for stability
N_EQ       = 500
EQ_ANGLES  = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]
VAL_DIR    = './imagenet_val'
OUT_PATH   = './results_imagenet/TAU_SWEEP_resnet50.json'


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    print(f'[DATA] {len(imgs)} images', flush=True)
    imgs = imgs[:N_INS_DEL]
    labs = labs[:N_INS_DEL]

    wrapper = BackboneWrapper('resnet50', device)
    labs = get_model_predictions(wrapper, imgs, labs)

    results = {}
    if os.path.exists(OUT_PATH):
        try:
            results = json.load(open(OUT_PATH))
            done = [k for k in results if 'eq' in (results.get(k) or {})]
            print(f'[RESUME] τ done: {done}', flush=True)
        except Exception:
            results = {}

    def save():
        tmp = OUT_PATH + '.tmp'
        with open(tmp, 'w') as f: json.dump(results, f, indent=2, default=str)
        os.replace(tmp, OUT_PATH)

    for tau in TAU_VALUES:
        key = f'tau_{tau:.2f}'
        if key in results and 'eq' in results[key]:
            print(f'  [{key}] SKIP', flush=True)
            continue
        print(f'\n  [{key}] CA² N=18 — {len(imgs)} ins/del images, {N_EQ} eq images', flush=True)
        method = CA2GradCAM_v2(wrapper, tau=tau, n_angles=18)
        call = _make_caller(method)
        ins_list, del_list = [], []
        t0 = time.time()
        for i, (img, cidx) in enumerate(zip(imgs, labs)):
            hm = call(img, cidx)
            ii, dd = insertion_deletion(wrapper, img, hm, cidx, 20, 10)
            ins_list.append(ii); del_list.append(dd)
            if (i+1) % 500 == 0:
                print(f'    {i+1}/{len(imgs)}  Ins={np.mean(ins_list):.3f}', flush=True)
        print(f'    Equivariance ({N_EQ} imgs)...', flush=True)
        eq_dict = eq_scores(call, imgs[:N_EQ], labs[:N_EQ], EQ_ANGLES)
        elapsed = time.time() - t0
        results[key] = {
            'tau':       float(tau),
            'n':         len(imgs),
            'n_eq':      N_EQ,
            'eq':        float(eq_dict['pearson']),
            'eq_std':    float(eq_dict['pearson_std']),
            'ins':       float(np.mean(ins_list)),
            'ins_std':   float(np.std(ins_list)),
            'del':       float(np.mean(del_list)),
            'del_std':   float(np.std(del_list)),
            'time_img':  elapsed / max(1, len(imgs)),
            'per_angle': eq_dict['per_angle'],
        }
        save()
        r = results[key]
        print(f'  [{key}] Eq={r["eq"]:.3f}±{r["eq_std"]:.3f} Ins={r["ins"]:.3f} Del={r["del"]:.3f}', flush=True)
    print(f'[SAVED] {OUT_PATH}', flush=True)


if __name__ == '__main__':
    main()
