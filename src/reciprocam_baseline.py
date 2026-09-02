"""Recipro-CAM baseline (Byun & Lee, CVPR Workshops 2024; arXiv:2209.14074v3).

Gradient-free saliency: for each spatial position (i, j) in the last-conv-block
activation A in R^{C x h x w}, build a masked activation that keeps A's values
at (i, j) and zeroes everywhere else, then forward that masked tensor through
the rest of the network. The saliency value at (i, j) is the resulting target-
class softmax probability.

  M[i, j] = softmax(g(A^{(i,j)}))[c]
  where A^{(i,j)}[c, i', j'] = A[c, i', j'] if (i', j') == (i, j) else 0
        g = the downstream-of-hook portion of the network.

The h*w masked-forwards are batched and run in one model call for speed.

Hook locations (independent of v11's wrapper; wrapper-external code):
  - resnet50:  model.layer4              -> A in (1, 2048, 7, 7); 49 positions
                downstream: avgpool -> flatten -> fc
  - vgg16:     model.features[29]        -> A in (1, 512, 14, 14); 196 positions
                downstream: features[30] (MaxPool) -> avgpool -> flatten ->
                            classifier
  - vit_b_16:  model.encoder.layers[-1]  -> A in (1, 197, 768); 196 patch
                positions (CLS token always kept). Downstream:
                encoder.ln -> select CLS -> heads.head

NEW FILE. Does NOT modify v11. Imports BackboneWrapper and helpers from
ca2_complete_eval.

Invocation:
  python reciprocam_baseline.py <resnet50|vgg16|vit_b_16> [n_eq=2000]
Output: results_imagenet_official/<backbone>__ReciproCAM.json
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
MASKED_BS  = 196   # batch size for masked-forward (=h*w when feasible)


class ReciproCAM:
    """Gradient-free per-spatial-position masked-forward saliency."""

    def __init__(self, wrapper: BackboneWrapper, masked_bs: int = MASKED_BS):
        self.w = wrapper
        self.bb = wrapper.name
        self.device = wrapper.device
        self.masked_bs = masked_bs
        self._A = None
        self._hook = self._install_hook()

    def _install_hook(self):
        if self.bb == 'resnet50':
            layer = self.w.model.layer4
        elif self.bb == 'vgg16':
            layer = self.w.model.features[29]
        elif self.bb == 'vit_b_16':
            # Hook the OUTPUT of layers[-2] (== input to layers[-1]). At
            # layers[-1] output, CLS has already aggregated every patch via
            # self-attention, so masking patches there cannot change CLS and
            # yields a constant heatmap. By hooking one block earlier, the
            # last block's self-attention re-mixes the masked tokens into
            # CLS in a position-dependent way.
            layer = self.w.model.encoder.layers[-2]
        else:
            raise ValueError(f'Unsupported backbone {self.bb}')

        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            self._A = o.detach()
        return layer.register_forward_hook(hook)

    def _downstream(self, A_masked):
        """Forward A_masked (shape B x ...) through layers after the hook."""
        m = self.w.model
        if self.bb == 'resnet50':
            x = m.avgpool(A_masked)
            x = torch.flatten(x, 1)
            return m.fc(x)
        elif self.bb == 'vgg16':
            x = m.features[30](A_masked)  # final MaxPool
            x = m.avgpool(x)
            x = torch.flatten(x, 1)
            return m.classifier(x)
        elif self.bb == 'vit_b_16':
            # A_masked: (B, 197, 768) — output of layers[-2]. Run the last
            # encoder block, the final LayerNorm, extract CLS, then heads.
            x = m.encoder.layers[-1](A_masked)
            x = m.encoder.ln(x)
            x = x[:, 0]
            return m.heads(x)

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    @torch.no_grad()
    def __call__(self, img, cidx):
        self.w.model.eval()
        H, W = img.shape[1], img.shape[2]
        x = img.unsqueeze(0).to(self.device)

        self._A = None
        _ = self.w.model(x)
        A = self._A  # detached by the hook
        if A is None:
            return np.zeros((H, W), dtype=np.float32)

        if self.bb in ('resnet50', 'vgg16'):
            # A: (1, C, h, w) -> build (h*w, C, h, w) one-hot masked copies.
            _, C, h, w = A.shape
            n = h * w
            # mask: (n, 1, h, w) with mask[k, 0, k//w, k%w] = 1
            mask = torch.zeros(n, 1, h, w, device=A.device)
            idx = torch.arange(n, device=A.device)
            mask[idx, 0, idx // w, idx % w] = 1.0
            scores = []
            for s in range(0, n, self.masked_bs):
                e = min(s + self.masked_bs, n)
                A_batch = A.expand(e - s, C, h, w) * mask[s:e]
                logits = self._downstream(A_batch)
                probs = F.softmax(logits, dim=1)
                scores.append(probs[:, cidx])
            sal = torch.cat(scores, dim=0).reshape(h, w)
        elif self.bb == 'vit_b_16':
            # A: (1, 197, 768) — patch tokens 1..196 are the 14x14 grid.
            B, T, D = A.shape
            n_patches = T - 1
            h = w = int(round(np.sqrt(n_patches)))
            assert h * w == n_patches, f'expected square patch grid, got {n_patches}'
            # mask: (n_patches, T) with index 0 (CLS) always 1, and exactly one
            # patch token at position k+1 also 1.
            mask = torch.zeros(n_patches, T, device=A.device)
            mask[:, 0] = 1.0
            mask[torch.arange(n_patches, device=A.device),
                 torch.arange(1, T, device=A.device)] = 1.0
            scores = []
            for s in range(0, n_patches, self.masked_bs):
                e = min(s + self.masked_bs, n_patches)
                A_batch = A.expand(e - s, T, D) * mask[s:e].unsqueeze(-1)
                logits = self._downstream(A_batch)
                probs = F.softmax(logits, dim=1)
                scores.append(probs[:, cidx])
            sal = torch.cat(scores, dim=0).reshape(h, w)
        else:
            return np.zeros((H, W), dtype=np.float32)

        sal = F.relu(sal)
        sal_up = F.interpolate(sal.unsqueeze(0).unsqueeze(0), (H, W),
                               mode='bilinear', align_corners=False).squeeze()
        mn, mx = sal_up.min(), sal_up.max()
        if (mx - mn).item() > 1e-8:
            sal_up = (sal_up - mn) / (mx - mn + 1e-8)
        return sal_up.cpu().numpy().astype(np.float32)


def per_image_eq(call, img, cidx, angles):
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
        print('usage: reciprocam_baseline.py <resnet50|vgg16|vit_b_16> [n_eq=2000]')
        sys.exit(2)
    bb = sys.argv[1]
    global N_EQ, N_INSDEL
    if len(sys.argv) >= 3:
        N_EQ = int(sys.argv[2])
        N_INSDEL = N_EQ
        print(f'[CFG] n_eq=n_insdel override = {N_EQ}', flush=True)

    out_path = f'./results_imagenet_official/{bb}__ReciproCAM.json'
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
    method = ReciproCAM(wrapper)

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
        if (i + 1) % 100 == 0:
            chunk_elapsed = time.time() - chunk_t0
            elapsed = time.time() - t0
            remaining = (N_INSDEL - i - 1) * (chunk_elapsed / 100)
            print(f'  [INS {i+1}/{N_INSDEL}] Ins={np.mean(state["ins_arr"]):.3f} '
                  f'Del={np.mean(state["del_arr"]):.3f} deg={state["degenerate_count"]} '
                  f'chunk={chunk_elapsed:.0f}s remaining={remaining/3600:.2f}h',
                  flush=True)
            save_state()
            chunk_t0 = time.time()
    save_state()

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
        if (i + 1) % 100 == 0:
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
        'config': {'backbone': bb, 'method': 'ReciproCAM', 'masked_bs': MASKED_BS,
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
