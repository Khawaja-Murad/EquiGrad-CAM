"""Single-method, single-backbone full-scale evaluation on official ILSVRC val.

Ins/del on 10K images + equivariance on 10K images × 7 angles. Per-image
checkpoint every 500 images so SLURM walltime kills lose <500 imgs of work.

Invocation:
  python run_official_one.py <backbone> <method>
  backbone ∈ {resnet50, vgg16, vit_b_16}
  method   ∈ {GradCAM, GradCAM++, XGrad-CAM, LayerCAM, SmoothGC++,
              ScoreCAM, AugCAM_t0, AugCAM_t01, CA2_N6, CA2_N18}

Output: results_imagenet_official/<backbone>__<safe_method>.json
"""
import os, sys, json, time
import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper,
    GradCAM_v2, GradCAMPP, XGradCAM, LayerCAM, ScoreCAM,
    SmoothGradCAMPP, AugCAM_v2, CA2GradCAM_v2,
    load_imagenet_val, get_model_predictions,
    insertion_deletion, rpbh_rotate, rotate_heatmap_np, _make_caller,
)

VAL_DIR   = './imagenet_val_official'
N_IMGS    = 10000
N_EQ      = 10000
CHUNK     = 500
EQ_ANGLES = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]


def _safe(name):
    return name.replace('+', 'p').replace('-', '_')


def build_method(name, wrapper):
    if name == 'GradCAM':    return GradCAM_v2(wrapper)
    if name == 'GradCAM++':  return GradCAMPP(wrapper)
    if name == 'XGrad-CAM':  return XGradCAM(wrapper)
    if name == 'LayerCAM':   return LayerCAM(wrapper)
    if name == 'SmoothGC++': return SmoothGradCAMPP(wrapper, n_samples=50)
    if name == 'ScoreCAM':   return ScoreCAM(wrapper, batch_size=64)
    if name == 'AugCAM_t0':  return AugCAM_v2(wrapper, tau=0.0)
    if name == 'AugCAM_t01': return AugCAM_v2(wrapper, tau=0.10)
    if name == 'CA2_N6':     return CA2GradCAM_v2(wrapper, tau=0.10, n_angles=6)
    if name == 'CA2_N18':    return CA2GradCAM_v2(wrapper, tau=0.10, n_angles=18)
    raise ValueError(f'unknown method {name}')


def per_image_eq_one(call, img, cidx, angles):
    """Return (mean_pearson_over_angles, per_angle_dict) or None."""
    try:
        hm0 = call(img, cidx)
    except Exception:
        return None
    if hm0.std() < 1e-8: return None
    pa = {}
    for a in angles:
        try:
            rot = rpbh_rotate(img, float(a))
            hm_r = call(rot, cidx)
        except Exception:
            continue
        if hm_r.std() < 1e-8: continue
        ref = rotate_heatmap_np(hm0, float(a))
        if ref.std() < 1e-8: continue
        try:
            p, _ = pearsonr(hm_r.flatten(), ref.flatten())
        except Exception:
            continue
        if np.isfinite(p):
            pa[a] = float(p)
    if not pa: return None
    return float(np.mean(list(pa.values()))), pa


def main():
    if len(sys.argv) < 3:
        print('usage: run_official_one.py <backbone> <method>')
        sys.exit(2)
    bb, method_name = sys.argv[1], sys.argv[2]
    if bb == 'vit_b_16' and method_name == 'ScoreCAM':
        print('[SKIP] ScoreCAM excluded for ViT'); sys.exit(0)

    out_dir = './results_imagenet_official'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{bb}__{_safe(method_name)}.json')

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    print(f'[DATA] {len(imgs)} OFFICIAL ImageNet val images', flush=True)
    imgs = imgs[:N_IMGS]
    labs = labs[:N_IMGS]

    wrapper = BackboneWrapper(bb, device)
    labs = get_model_predictions(wrapper, imgs, labs)

    method = build_method(method_name, wrapper)
    call = _make_caller(method)

    # Resume state
    state = {
        'ins': [], 'dele': [],
        'per_image_eq': [], 'per_angle_running': {str(a): [] for a in EQ_ANGLES},
        'i_ins_next': 0, 'i_eq_next': 0,
    }
    if os.path.exists(out_path):
        try:
            state = json.load(open(out_path))
            print(f'[RESUME] {out_path}: ins-next={state["i_ins_next"]}, eq-next={state["i_eq_next"]}', flush=True)
        except Exception as e:
            print(f'[RESUME] failed: {e}', flush=True)

    def save_state():
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, out_path)

    # Phase 1: ins/del on 10K
    t0 = time.time()
    chunk_t0 = time.time()
    for i in range(state['i_ins_next'], N_IMGS):
        try:
            hm = call(imgs[i], labs[i])
            ii, dd = insertion_deletion(wrapper, imgs[i], hm, labs[i], 20, 10)
            state['ins'].append(ii); state['dele'].append(dd)
        except Exception as e:
            print(f'  [INS/DEL {i}] error: {e}', flush=True)
        state['i_ins_next'] = i + 1
        if (i + 1) % CHUNK == 0:
            elapsed = time.time() - t0
            chunk = time.time() - chunk_t0
            print(f'  [INS/DEL {i+1}/{N_IMGS}] Ins={np.mean(state["ins"]):.3f}  '
                  f'chunk={chunk:.0f}s, elapsed={elapsed/3600:.2f}h', flush=True)
            save_state()
            chunk_t0 = time.time()

    save_state()

    # Phase 2: equivariance on 10K
    chunk_t0 = time.time()
    for i in range(state['i_eq_next'], N_EQ):
        result = per_image_eq_one(call, imgs[i], labs[i], EQ_ANGLES)
        if result is not None:
            score, pa = result
            state['per_image_eq'].append(score)
            for a, p in pa.items():
                state['per_angle_running'][str(a)].append(p)
        state['i_eq_next'] = i + 1
        if (i + 1) % CHUNK == 0:
            elapsed = time.time() - t0
            chunk = time.time() - chunk_t0
            running_eq = float(np.mean(state['per_image_eq'])) if state['per_image_eq'] else 0
            est_remaining = (N_EQ - i - 1) * chunk / CHUNK
            print(f'  [EQ {i+1}/{N_EQ}] running Eq={running_eq:.3f}  '
                  f'chunk={chunk:.0f}s, elapsed={elapsed/3600:.2f}h, est_remaining={est_remaining/3600:.2f}h',
                  flush=True)
            save_state()
            chunk_t0 = time.time()

    # Phase 3: summary
    ins_arr = np.array(state['ins'], dtype=float)
    del_arr = np.array(state['dele'], dtype=float)
    eq_arr = np.array(state['per_image_eq'], dtype=float)
    per_angle = {a: float(np.mean(v)) if v else 0.0 for a, v in state['per_angle_running'].items()}
    summary = {
        'config': {
            'backbone': bb, 'method': method_name, 'val_dir': VAL_DIR,
            'n': N_IMGS, 'n_eq': N_EQ, 'eq_angles': EQ_ANGLES, 'seed': 42,
        },
        'result': {
            'n':       int(ins_arr.size),
            'n_eq':    int(eq_arr.size),
            'eq':      float(eq_arr.mean()) if eq_arr.size else 0.0,
            'eq_std':  float(eq_arr.std())  if eq_arr.size else 0.0,
            'ins':     float(ins_arr.mean()) if ins_arr.size else 0.0,
            'ins_std': float(ins_arr.std())  if ins_arr.size else 0.0,
            'del':     float(del_arr.mean()) if del_arr.size else 0.0,
            'del_std': float(del_arr.std())  if del_arr.size else 0.0,
            'time_total': time.time() - t0,
            'per_angle': per_angle,
        },
        # keep raw arrays for downstream significance / re-aggregation
        'ins_arr': state['ins'],
        'del_arr': state['dele'],
        'per_image_eq': state['per_image_eq'],
        'per_angle_running': state['per_angle_running'],
        'i_ins_next': state['i_ins_next'],
        'i_eq_next': state['i_eq_next'],
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    r = summary['result']
    print(f'\n[SAVED] {out_path}', flush=True)
    print(f'  Eq={r["eq"]:.3f}±{r["eq_std"]:.3f}  '
          f'Ins={r["ins"]:.3f}±{r["ins_std"]:.3f}  '
          f'Del={r["del"]:.3f}±{r["del_std"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
