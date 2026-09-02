"""Job C revised — Equal-compute comparison on ResNet-50 only.

Runs RISE-400-masks and a top-K=80 channel variant of Score-CAM at the
EquiGrad-CAM T=18 wall-clock budget (~0.55 s/img on A100), n_eq=10K full
ImageNet-1K val.

Implementation note: top-80 Score-CAM uses the first 80 channels by GAP score
descending (same convention as paper-v1's truncated Score-CAM sub-runs).
RISE-400 is the same RISE class with N_MASKS=400.

NEW FILE. Imports BackboneWrapper/ScoreCAM from ca2_complete_eval but does
not modify v11.

Invocation:
  python equal_compute_runner.py <method>
  method in {'rise400', 'scorecam80'}

Output: results_imagenet_official/resnet50__{rise400|scorecam80}.json
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper, ScoreCAM,
    load_imagenet_val, get_model_predictions,
    insertion_deletion, rpbh_rotate, rotate_heatmap_np, _make_caller,
)
from rise_baseline import RISE, per_image_eq

VAL_DIR = './imagenet_val_official'
N_INSDEL = 10000
N_EQ     = 10000
EQ_ANGLES = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]
BB = 'resnet50'


class ScoreCAM_TopK:
    """Score-CAM truncated to the top-K channels by GAP score."""
    def __init__(self, wrapper: BackboneWrapper, top_k: int = 80, batch_size: int = 64):
        self.wrapper = wrapper
        self.top_k = top_k
        self.sc = ScoreCAM(wrapper, batch_size=batch_size)

    def __call__(self, img: torch.Tensor, cidx: int):
        """Get activations + run ScoreCAM with channel limit."""
        # Use the v11 wrapper's built-in forward hook (it already captures
        # the last conv-block activation into self.wrapper._acts).
        device = next(self.wrapper.model.parameters()).device
        x = img.to(device).unsqueeze(0)
        self.wrapper.model.eval()
        with torch.no_grad():
            _ = self.wrapper.model(x)
        A = self.wrapper._acts[0].detach()
        if A.dim() == 4:
            A = A[0]
        C, h, w = A.shape
        gap = A.mean(dim=(1, 2))  # C
        topk_idx = torch.topk(gap.abs(), k=min(self.top_k, C)).indices

        # Now run channel masking only on those top-K channels
        masks = []
        for c in topk_idx.tolist():
            mc = A[c]
            # Upsample to 224x224 and normalise
            up = F.interpolate(mc.unsqueeze(0).unsqueeze(0), size=(224, 224),
                                 mode='bilinear', align_corners=False)[0, 0]
            mn, mx = up.min(), up.max()
            if mx > mn:
                up = (up - mn) / (mx - mn + 1e-8)
            else:
                up = torch.zeros_like(up)
            masks.append(up)
        masks = torch.stack(masks, dim=0)  # K,224,224

        # Batched forward
        scores = []
        with torch.no_grad():
            for i in range(0, masks.shape[0], 32):
                j = min(i + 32, masks.shape[0])
                m = masks[i:j].unsqueeze(1)  # B,1,H,W
                masked = x * m
                logits = self.wrapper.model(masked)
                probs = F.softmax(logits, dim=1)[:, cidx]
                scores.append(probs)
        scores = torch.cat(scores)  # K

        # Heatmap = sum_k(score_k * upsampled_mask_k)
        # but use the original A-channel maps not the upsampled normalised ones for the weighting
        # Standard ScoreCAM: heatmap_full = ReLU(sum_k score_k * A_k_upsampled)
        hm = torch.zeros(224, 224, device=device)
        for ki, c in enumerate(topk_idx.tolist()):
            mc = A[c]
            up = F.interpolate(mc.unsqueeze(0).unsqueeze(0), size=(224, 224),
                               mode='bilinear', align_corners=False)[0, 0]
            hm = hm + scores[ki] * up
        hm = F.relu(hm)
        return hm.cpu().numpy()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('rise400', 'scorecam80'):
        print('usage: equal_compute_runner.py <rise400|scorecam80>')
        sys.exit(2)
    method = sys.argv[1]
    out = f'./results_imagenet_official/{BB}__{method}.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    n_use = max(N_INSDEL, N_EQ)
    imgs = imgs[:n_use]; labs = labs[:n_use]
    print(f'[DATA] {len(imgs)} images', flush=True)

    wrapper = BackboneWrapper(BB, device)
    labs = get_model_predictions(wrapper, imgs, labs)

    if method == 'rise400':
        m = RISE(wrapper, n_masks=400, batch=128, device=device)
    else:
        m = ScoreCAM_TopK(wrapper, top_k=80, batch_size=64)

    # Resume support
    data = {}
    if os.path.exists(out):
        try: data = json.load(open(out))
        except: data = {}
    ins_arr = data.get('ins_arr', [])
    del_arr = data.get('del_arr', [])
    eq_arr  = data.get('per_image_eq', [])
    i_ins_next = data.get('i_ins_next', len(ins_arr))
    i_eq_next  = data.get('i_eq_next', len(eq_arr))

    def save():
        tmp = out + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({
                'config': {'backbone': BB, 'method': method, 'n_insdel': N_INSDEL, 'n_eq': N_EQ},
                'ins_arr': ins_arr, 'del_arr': del_arr, 'per_image_eq': eq_arr,
                'i_ins_next': i_ins_next, 'i_eq_next': i_eq_next,
            }, f)
        os.replace(tmp, out)

    t0 = time.time()
    for i in range(i_ins_next, N_INSDEL):
        try:
            hm = m(imgs[i], int(labs[i]))
        except Exception as e:
            print(f'  [INS {i}] EXC {e}', flush=True); continue
        ii, dd = insertion_deletion(wrapper, imgs[i], hm, int(labs[i]), 20, 10)
        ins_arr.append(float(ii)); del_arr.append(float(dd))
        i_ins_next = i + 1
        if (i + 1) % 200 == 0:
            save()
            elapsed = (time.time() - t0)
            print(f'  [INS {i+1}/{N_INSDEL}] Ins={np.mean(ins_arr):.3f} Del={np.mean(del_arr):.3f}  '
                  f'elapsed={elapsed/3600:.2f}h', flush=True)

    t1 = time.time()
    for i in range(i_eq_next, N_EQ):
        eq = per_image_eq(m, imgs[i], int(labs[i]), EQ_ANGLES)
        if eq is not None: eq_arr.append(eq)
        i_eq_next = i + 1
        if (i + 1) % 200 == 0:
            save()
            print(f'  [EQ {i+1}/{N_EQ}] Eq={np.mean(eq_arr):.3f}  chunk_elapsed={(time.time()-t1):.0f}s', flush=True)

    result = {
        'n': len(ins_arr), 'n_eq': len(eq_arr),
        'eq': float(np.mean(eq_arr)) if eq_arr else None,
        'eq_std': float(np.std(eq_arr)) if eq_arr else None,
        'ins': float(np.mean(ins_arr)) if ins_arr else None,
        'del': float(np.mean(del_arr)) if del_arr else None,
    }
    data = {'config': {'backbone': BB, 'method': method, 'n_insdel': N_INSDEL, 'n_eq': N_EQ},
            'result': result,
            'ins_arr': ins_arr, 'del_arr': del_arr, 'per_image_eq': eq_arr,
            'i_ins_next': i_ins_next, 'i_eq_next': i_eq_next}
    with open(out, 'w') as f: json.dump(data, f, indent=2, default=str)
    print(f'\n[SAVED] {out}', flush=True)
    print(f'  Eq={result["eq"]:.3f}  Ins={result["ins"]:.3f}  Del={result["del"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
