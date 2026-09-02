"""ViT-native baselines at full n_eq=10,000 on official ImageNet-1K val.

Re-runs AttentionRollout and TransformerAttribution under the v3 protocol
(7 angles, per-image Pearson, full 10K) so Table 4 aligns with Table 3.

Invocation:
  python run_vit_baselines_eq10k.py <method>
where method ∈ {AttentionRollout, TransformerAttribution}.

Output: results_imagenet_official/vit_b_16__<safe_method>.json
"""
import os, sys, json, time
import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed,
    ViTAttentionRecorder, AttentionRollout, TransformerAttribution,
    load_imagenet_val, get_model_predictions,
    insertion_deletion, rpbh_rotate, rotate_heatmap_np, _make_caller,
)
import torchvision.models as models

VAL_DIR   = './imagenet_val_official'
N_IMGS    = 10000
N_EQ      = 10000
CHUNK     = 500
EQ_ANGLES = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]


def _safe(name):
    return name.replace('+', 'p').replace('-', '_')


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
        print('usage: run_vit_baselines_eq10k.py <AttentionRollout|TransformerAttribution>')
        sys.exit(2)
    method_name = sys.argv[1]
    assert method_name in ('AttentionRollout', 'TransformerAttribution')

    safe = _safe(method_name)
    out_path = f'./results_imagenet_official/vit_b_16__{safe}.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    print(f'[DATA] {len(imgs)} images', flush=True)
    assert len(imgs) >= max(N_IMGS, N_EQ)
    imgs = imgs[:max(N_IMGS, N_EQ)]
    labs = labs[:max(N_IMGS, N_EQ)]

    # Build recorder + method
    vit = models.vit_b_16(weights='IMAGENET1K_V1').to(device).eval()
    recorder = ViTAttentionRecorder(vit, device=device)
    if method_name == 'AttentionRollout':
        method = AttentionRollout(recorder)
    else:
        method = TransformerAttribution(recorder)
    call = _make_caller(method)

    labs_pred = get_model_predictions(recorder, imgs, labs)

    # Resume state
    state = {'per_image_eq': [], 'per_angle_running': {str(a): [] for a in EQ_ANGLES},
             'ins_arr': [], 'del_arr': [], 'i_ins_next': 0, 'i_eq_next': 0}
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
            if 'result' in old and 'eq' in old.get('result', {}):
                print(f'[RESUME] {out_path} already DONE; nothing to do.', flush=True)
                return
            state.update({k: old.get(k, state[k]) for k in state})
            print(f'[RESUME] {out_path}: ins-next={state["i_ins_next"]}, eq-next={state["i_eq_next"]}', flush=True)
        except Exception as e:
            print(f'[RESUME] failed: {e}', flush=True)

    def save_state():
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f: json.dump(state, f, indent=2, default=str)
        os.replace(tmp, out_path)

    t0 = time.time()

    # Ins/Del pass
    chunk_t0 = time.time()
    for i in range(state['i_ins_next'], N_IMGS):
        try:
            hm = call(imgs[i], labs_pred[i])
        except Exception:
            state['i_ins_next'] = i + 1
            continue
        ii, dd = insertion_deletion(recorder, imgs[i], hm, labs_pred[i], 20, 10)
        state['ins_arr'].append(float(ii))
        state['del_arr'].append(float(dd))
        state['i_ins_next'] = i + 1
        if (i+1) % CHUNK == 0:
            print(f'  [INS {i+1}/{N_IMGS}] Ins={np.mean(state["ins_arr"]):.3f}  chunk={time.time()-chunk_t0:.0f}s', flush=True)
            save_state()
            chunk_t0 = time.time()
    save_state()

    # Eq pass
    chunk_t0 = time.time()
    for i in range(state['i_eq_next'], N_EQ):
        result = per_image_eq_one(call, imgs[i], labs_pred[i], EQ_ANGLES)
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
        'config': {'backbone': 'vit_b_16', 'method': method_name, 'n': N_IMGS, 'n_eq': N_EQ,
                   'eq_angles': EQ_ANGLES, 'seed': 42},
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
