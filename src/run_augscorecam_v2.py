"""Augmented Score-CAM baseline (Ibrahim & Shafiq 2022, ref [12] in paper).

Equivalent of AugCAM but with Score-CAM as the inner method instead of Grad-CAM:
per rotated view, run Score-CAM, inverse-rotate the heatmap, average in pixel
space (NO PCF, NO confidence weighting --- the canonical pixel-space baseline).

Cost is the dominant constraint: Score-CAM is ~2 s/img on A100, and we evaluate
at N=6 rotations per call (full N=18 is intractable at n=10K). Final t/img is
roughly 12 s/Eq-call * 8 calls/image = 96 s/image, so for n=10K Eq alone the
walltime budget would be ~270 h --- not feasible.

We therefore run AugScoreCAM at the same protocol used for the partial
Score-CAM cells in paper v2: full n=10000 ins/del, n_eq=500 equivariance,
N=6 inner rotations.

Invocation:
  python run_augscorecam.py <backbone>
  backbone ∈ {resnet50, vgg16}  (ScoreCAM skipped for ViT)

Output: results_imagenet_official/<backbone>__AugScoreCAM.json
"""
import os, sys, json, time
import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper, ScoreCAM,
    load_imagenet_val, get_model_predictions,
    insertion_deletion, rpbh_rotate, rotate_heatmap_np, _make_caller,
)

VAL_DIR    = './imagenet_val_official'
N_IMGS     = 10000   # ins/del sample (overridable via CLI)
N_EQ       = 500     # equivariance sample (overridable via CLI)
N_ROT      = 6
CHUNK      = 100
EQ_ANGLES  = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]


class AugScoreCAM:
    """Per rotated view, Score-CAM; inverse-rotate; average in pixel space."""
    def __init__(self, wrapper, n_rot=6, pad=64):
        self.w = wrapper
        self.angles = list(np.linspace(-180, 180, n_rot, endpoint=False))
        self.pad = pad
        self.sc = ScoreCAM(wrapper, batch_size=64)
    def __call__(self, img, cidx):
        H, W = img.shape[1], img.shape[2]
        maps = []
        for a in self.angles:
            rot = rpbh_rotate(img, float(a), self.pad)
            try:
                hm_r = self.sc(rot, cidx)
            except Exception:
                continue
            # Inverse-rotate the heatmap back to original orientation
            hm_back = rotate_heatmap_np(hm_r, -float(a))
            maps.append(hm_back)
        if not maps:
            return np.zeros((H, W), dtype=np.float32)
        agg = np.mean(maps, axis=0)
        agg = np.maximum(agg, 0)
        mn, mx = agg.min(), agg.max()
        if mx > mn:
            agg = (agg - mn) / (mx - mn)
        return agg


def per_image_eq_one(call, img, cidx, angles):
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
    if len(sys.argv) < 2:
        print('usage: run_augscorecam_v2.py <resnet50|vgg16> [n_eq=500]')
        sys.exit(2)
    bb = sys.argv[1]
    assert bb in ('resnet50', 'vgg16'), 'AugScoreCAM skipped for ViT (no ScoreCAM)'

    global N_EQ
    if len(sys.argv) >= 3:
        N_EQ = int(sys.argv[2])
        print(f'[CFG] n_eq override = {N_EQ}', flush=True)

    out_path = f'./results_imagenet_official/{bb}__AugScoreCAM_v2.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    print(f'[DATA] {len(imgs)} images', flush=True)
    imgs = imgs[:N_IMGS]; labs = labs[:N_IMGS]

    wrapper = BackboneWrapper(bb, device)
    labs = get_model_predictions(wrapper, imgs, labs)

    method = AugScoreCAM(wrapper, n_rot=N_ROT)
    call = _make_caller(method)

    state = {'per_image_eq': [], 'per_angle_running': {str(a): [] for a in EQ_ANGLES},
             'ins_arr': [], 'del_arr': [], 'i_ins_next': 0, 'i_eq_next': 0}
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
            if 'result' in old and old['result'].get('eq') is not None:
                print(f'[RESUME] {out_path} already DONE.', flush=True)
                return
            state.update({k: old.get(k, state[k]) for k in state})
            print(f'[RESUME] ins-next={state["i_ins_next"]}, eq-next={state["i_eq_next"]}', flush=True)
        except Exception as e:
            print(f'[RESUME] failed: {e}', flush=True)

    def save_state():
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f: json.dump(state, f, indent=2, default=str)
        os.replace(tmp, out_path)

    t0 = time.time()

    # Ins/Del on 10K
    chunk_t0 = time.time()
    for i in range(state['i_ins_next'], N_IMGS):
        try:
            hm = call(imgs[i], labs[i])
        except Exception as e:
            state['i_ins_next'] = i + 1
            continue
        ii, dd = insertion_deletion(wrapper, imgs[i], hm, labs[i], 20, 10)
        state['ins_arr'].append(float(ii))
        state['del_arr'].append(float(dd))
        state['i_ins_next'] = i + 1
        if (i+1) % CHUNK == 0:
            elapsed = time.time() - t0
            chunk_elapsed = time.time() - chunk_t0
            remaining = (N_IMGS - i - 1) * chunk_elapsed / CHUNK
            print(f'  [INS {i+1}/{N_IMGS}] Ins={np.mean(state["ins_arr"]):.3f}  chunk={chunk_elapsed:.0f}s elapsed={elapsed/3600:.2f}h remaining={remaining/3600:.2f}h', flush=True)
            save_state()
            chunk_t0 = time.time()
    save_state()

    # Eq on 500
    chunk_t0 = time.time()
    for i in range(state['i_eq_next'], N_EQ):
        result = per_image_eq_one(call, imgs[i], labs[i], EQ_ANGLES)
        if result is None:
            state['i_eq_next'] = i + 1
            continue
        score, pa = result
        state['per_image_eq'].append(score)
        for a, p in pa.items():
            state['per_angle_running'][str(a)].append(p)
        state['i_eq_next'] = i + 1
        if (i+1) % CHUNK == 0:
            cur = float(np.mean(state['per_image_eq']))
            print(f'  [EQ {i+1}/{N_EQ}] running Eq={cur:.3f}  chunk={time.time()-chunk_t0:.0f}s', flush=True)
            save_state()
            chunk_t0 = time.time()

    arr = np.array(state['per_image_eq'], dtype=float)
    ins = np.array(state['ins_arr'], dtype=float)
    dee = np.array(state['del_arr'], dtype=float)
    summary = {
        'config': {'backbone': bb, 'method': 'AugScoreCAM', 'n_rot_inner': N_ROT,
                   'n': N_IMGS, 'n_eq': N_EQ, 'eq_angles': EQ_ANGLES, 'seed': 42},
        'result': {
            'n': int(ins.size), 'n_eq': int(arr.size),
            'eq': float(arr.mean()) if arr.size else 0.0,
            'eq_std': float(arr.std()) if arr.size else 0.0,
            'ins': float(ins.mean()) if ins.size else 0.0,
            'ins_std': float(ins.std()) if ins.size else 0.0,
            'del': float(dee.mean()) if dee.size else 0.0,
            'del_std': float(dee.std()) if dee.size else 0.0,
            'time_total': time.time() - t0,
            'per_angle': {a: float(np.mean(v)) if v else 0.0 for a, v in state['per_angle_running'].items()},
        },
        'per_image_eq': state['per_image_eq'],
        'ins_arr': state['ins_arr'],
        'del_arr': state['del_arr'],
        'per_angle_running': state['per_angle_running'],
        'i_ins_next': N_IMGS, 'i_eq_next': N_EQ,
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'[SAVED] {out_path}  Eq={summary["result"]["eq"]:.3f}±{summary["result"]["eq_std"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
