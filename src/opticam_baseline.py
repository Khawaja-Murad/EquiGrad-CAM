"""Opti-CAM baseline (Zhang et al., CVIU 2024; arXiv:2301.07002).

Per-image optimisation of softmax-normalised channel weights to maximise the
target-class logit on a masked-image forward.

Algorithm (as implemented here, adapted from the paper's description):
  1.  One forward pass through the frozen backbone gives feature activations
      A in R^{1 x C x h x w} at the v11 hook layer (matching every other
      baseline). For ViT we reshape patch tokens 196 -> (14, 14, D).
  2.  Initialise beta = 0 in R^C; let u = softmax(beta) so the initial weights
      are uniform 1/C.
  3.  For N iterations:
        S = sum_k u_k * A_k                                     in R^{h x w}
        S_up = bilinear upsample to (H, W) and min-max normalise to [0, 1]
        masked = img * S_up
        loss = -model(masked)[target_class]   (maximise the target logit)
        Adam step on beta only (backbone is frozen).
  4.  Final heatmap = ReLU(sum_k u*_k A_k) upsampled to (H, W), min-max
      normalised to [0, 1].

Hyperparameters: N = 100 iterations, Adam lr = 0.1, target = raw logit
(maximising softmax probability would saturate). These match the paper's
defaults; no per-backbone tuning.

NEW FILE. Does NOT modify v11. Imports BackboneWrapper and helpers from
ca2_complete_eval.

Invocation:
  python opticam_baseline.py <resnet50|vgg16|vit_b_16> [n_eq=2000]
Output: results_imagenet_official/<backbone>__OptiCAM.json
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper,
    load_imagenet_val, get_model_predictions,
    insertion_deletion, rpbh_rotate, rotate_heatmap_np,
)

VAL_DIR    = './imagenet_val_official'
N_INSDEL   = 2000
N_EQ       = 2000
EQ_ANGLES  = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]
N_ITERS    = 100
LR         = 0.1


class OptiCAM:
    """Per-image softmax-channel-weight optimisation (Zhang et al., CVIU 2024)."""

    def __init__(self, wrapper: BackboneWrapper, n_iters=N_ITERS, lr=LR):
        self.w = wrapper
        self.n_iters = n_iters
        self.lr = lr

    def _get_activations(self, x):
        """One forward to populate wrapper's stored activation tensor."""
        with torch.no_grad():
            _ = self.w.model(x)
        A = self.w._acts[0]  # already detached by the wrapper's hook
        if self.w.is_vit:
            # Drop CLS token (if present) and reshape patches to (1, D, 14, 14).
            if A.dim() == 3:
                A = A[0]
            if A.dim() == 2:
                if A.shape[0] == 197:
                    A = A[1:]
                D = A.shape[1]
                h = w = int(A.shape[0] ** 0.5)
                A = A.permute(1, 0).reshape(D, h, w).unsqueeze(0)
        if A.dim() == 3:  # (C, h, w) -> (1, C, h, w)
            A = A.unsqueeze(0)
        return A.contiguous()

    def __call__(self, img, cidx):
        self.w.model.eval()
        H, W = img.shape[1], img.shape[2]
        device = self.w.device
        x = img.unsqueeze(0).to(device)

        A = self._get_activations(x).to(device).detach()
        _, C, hh, ww = A.shape

        # Freeze backbone parameters (eval() already disables dropout/BN updates,
        # but we must also prevent autograd from touching model weights).
        for p in self.w.model.parameters():
            p.requires_grad_(False)

        beta = torch.zeros(C, device=device, requires_grad=True)
        opt = torch.optim.Adam([beta], lr=self.lr)

        for _ in range(self.n_iters):
            opt.zero_grad()
            u = torch.softmax(beta, dim=0)  # (C,)
            S = (u[None, :, None, None] * A).sum(dim=1, keepdim=True)  # (1,1,h,w)
            S_up = F.interpolate(S, (H, W), mode='bilinear', align_corners=False)
            mn, mx = S_up.min(), S_up.max()
            if (mx - mn).item() > 1e-8:
                S_up = (S_up - mn) / (mx - mn + 1e-8)
            else:
                # all-zero S -> nothing to mask; abort the optimisation early
                break
            masked = x * S_up
            logits = self.w.model(masked)
            loss = -logits[0, cidx]
            loss.backward()
            opt.step()

        with torch.no_grad():
            u = torch.softmax(beta, dim=0)
            S = (u[None, :, None, None] * A).sum(dim=1).squeeze(0)  # (h, w)
            S = F.relu(S)
            S = F.interpolate(S.unsqueeze(0).unsqueeze(0), (H, W),
                              mode='bilinear', align_corners=False).squeeze()
            mn, mx = S.min(), S.max()
            if (mx - mn).item() > 1e-8:
                S = (S - mn) / (mx - mn + 1e-8)
        return S.detach().cpu().numpy().astype(np.float32)


def per_image_eq(call, img, cidx, angles):
    """Mean Pearson over angles. Returns (mean, per_angle_dict) or None."""
    try:
        hm0 = call(img, cidx)
    except Exception:
        return None
    if hm0.std() < 1e-8:
        return None
    pa = {}
    for a in angles:
        try:
            rot = rpbh_rotate(img, float(a))
            hm_r = call(rot, cidx)
        except Exception:
            continue
        if hm_r.std() < 1e-8:
            continue
        ref = rotate_heatmap_np(hm0, float(a))
        if ref.std() < 1e-8:
            continue
        try:
            p, _ = pearsonr(hm_r.flatten(), ref.flatten())
        except Exception:
            continue
        if np.isfinite(p):
            pa[float(a)] = float(p)
    if not pa:
        return None
    return float(np.mean(list(pa.values()))), pa


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('resnet50', 'vgg16', 'vit_b_16'):
        print('usage: opticam_baseline.py <resnet50|vgg16|vit_b_16> [n_eq=2000]')
        sys.exit(2)
    bb = sys.argv[1]
    global N_EQ, N_INSDEL
    if len(sys.argv) >= 3:
        N_EQ = int(sys.argv[2])
        N_INSDEL = N_EQ
        print(f'[CFG] n_eq=n_insdel override = {N_EQ}', flush=True)

    out_path = f'./results_imagenet_official/{bb}__OptiCAM.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    n_use = max(N_INSDEL, N_EQ)
    imgs = imgs[:n_use]
    labs = labs[:n_use]
    print(f'[DATA] {len(imgs)} images for {bb}', flush=True)

    wrapper = BackboneWrapper(bb, device)
    labs = get_model_predictions(wrapper, imgs, labs)
    method = OptiCAM(wrapper, n_iters=N_ITERS, lr=LR)

    def call(img, cidx):
        return method(img, cidx)

    state = {'per_image_eq': [], 'per_angle_running': {str(a): [] for a in EQ_ANGLES},
             'ins_arr': [], 'del_arr': [], 'i_ins_next': 0, 'i_eq_next': 0,
             'degenerate_count': 0}
    if os.path.exists(out_path):
        try:
            old = json.load(open(out_path))
            if 'result' in old and old['result'].get('eq') is not None:
                print(f'[RESUME] {out_path} already DONE', flush=True)
                return
            for k in state:
                state[k] = old.get(k, state[k])
            print(f'[RESUME] ins-next={state["i_ins_next"]}, eq-next={state["i_eq_next"]}',
                  flush=True)
        except Exception as e:
            print(f'[RESUME] failed: {e}', flush=True)

    def save_state():
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, out_path)

    t0 = time.time()
    # Ins/Del pass
    chunk_t0 = time.time()
    for i in range(state['i_ins_next'], min(N_INSDEL, len(imgs))):
        try:
            hm = call(imgs[i], int(labs[i]))
        except Exception as e:
            state['i_ins_next'] = i + 1
            print(f'  [INS {i}] EXCEPT {e}', flush=True)
            continue
        if hm.std() < 1e-8:
            state['degenerate_count'] += 1
        ii, dd = insertion_deletion(wrapper, imgs[i], hm, int(labs[i]), 20, 10)
        state['ins_arr'].append(float(ii))
        state['del_arr'].append(float(dd))
        state['i_ins_next'] = i + 1
        if (i + 1) % 50 == 0:
            chunk_elapsed = time.time() - chunk_t0
            elapsed = time.time() - t0
            remaining = (N_INSDEL - i - 1) * (chunk_elapsed / 50)
            print(f'  [INS {i+1}/{N_INSDEL}] Ins={np.mean(state["ins_arr"]):.3f} '
                  f'Del={np.mean(state["del_arr"]):.3f} deg={state["degenerate_count"]} '
                  f'chunk={chunk_elapsed:.0f}s remaining={remaining/3600:.2f}h',
                  flush=True)
            save_state()
            chunk_t0 = time.time()
    save_state()

    # Eq pass
    print(f'\n[EQ] starting on n_eq={N_EQ}', flush=True)
    chunk_t0 = time.time()
    for i in range(state['i_eq_next'], min(N_EQ, len(imgs))):
        result = per_image_eq(call, imgs[i], int(labs[i]), EQ_ANGLES)
        if result is not None:
            score, pa = result
            state['per_image_eq'].append(score)
            for a, p in pa.items():
                state['per_angle_running'][str(float(a))].append(p)
        state['i_eq_next'] = i + 1
        if (i + 1) % 50 == 0:
            cur = float(np.mean(state['per_image_eq'])) if state['per_image_eq'] else 0.0
            chunk_elapsed = time.time() - chunk_t0
            print(f'  [EQ {i+1}/{N_EQ}] running Eq={cur:.3f} '
                  f'chunk={chunk_elapsed:.0f}s', flush=True)
            save_state()
            chunk_t0 = time.time()

    arr = np.array(state['per_image_eq'], dtype=float)
    ins = np.array(state['ins_arr'], dtype=float)
    dee = np.array(state['del_arr'], dtype=float)
    summary = {
        'config': {'backbone': bb, 'method': 'OptiCAM',
                   'n_iters': N_ITERS, 'lr': LR,
                   'n': N_INSDEL, 'n_eq': N_EQ, 'eq_angles': EQ_ANGLES, 'seed': 42},
        'result': {
            'n': int(ins.size), 'n_eq': int(arr.size),
            'eq': float(arr.mean()) if arr.size else 0.0,
            'eq_std': float(arr.std()) if arr.size else 0.0,
            'ins': float(ins.mean()) if ins.size else 0.0,
            'ins_std': float(ins.std()) if ins.size else 0.0,
            'del': float(dee.mean()) if dee.size else 0.0,
            'del_std': float(dee.std()) if dee.size else 0.0,
            'time_total': time.time() - t0,
            'degenerate_frac': state['degenerate_count'] / max(1, int(ins.size)),
            'per_angle': {a: float(np.mean(v)) if v else 0.0
                          for a, v in state['per_angle_running'].items()},
        },
        'per_image_eq': state['per_image_eq'],
        'ins_arr': state['ins_arr'],
        'del_arr': state['del_arr'],
        'per_angle_running': state['per_angle_running'],
        'i_ins_next': state['i_ins_next'],
        'i_eq_next': state['i_eq_next'],
    }
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f'[SAVED] {out_path} Eq={summary["result"]["eq"]:.3f}'
          f' Ins={summary["result"]["ins"]:.3f} Del={summary["result"]["del"]:.3f}',
          flush=True)


if __name__ == '__main__':
    main()
