"""Job A revised — RISE baseline (Petsiuk et al. 2018) at n_eq=2,000.

Standard RISE protocol:
  - N=8000 random masks, p=0.5, 7x7 grid upsampled bilinearly to 224x224
  - Saliency = sum_n(mask_n * score_n) / E[mask], where score is class prob
  - Batched mask evaluation on A100

NEW FILE. Does NOT modify v11. Imports BackboneWrapper from ca2_complete_eval.

Invocation:
  python rise_baseline.py <backbone>
where backbone in {resnet50, vgg16}. ViT not supported in this run (per scope).

Output: results_imagenet_official/<backbone>__RISE.json
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from skimage.transform import resize

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper,
    load_imagenet_val, get_model_predictions,
    insertion_deletion, rpbh_rotate, rotate_heatmap_np,
)

VAL_DIR = './imagenet_val_official'
N_INSDEL = 2000
N_EQ     = 2000
EQ_ANGLES = [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0]
N_MASKS  = 8000
MASK_GRID = 7
MASK_P   = 0.5
BATCH    = 128
IMG_SIZE = 224


def generate_masks(n_masks: int, grid_size: int, p: float, img_size: int,
                   device: torch.device, seed: int = 42):
    """Generate RISE random masks: small p-Bernoulli grid upsampled+cropped."""
    rng = np.random.default_rng(seed)
    cell = img_size // grid_size + 1
    up_size = (grid_size + 1) * cell
    masks = np.empty((n_masks, img_size, img_size), dtype=np.float32)
    bern = rng.random(size=(n_masks, grid_size, grid_size)).astype(np.float32) < p
    bern = bern.astype(np.float32)
    for i in range(n_masks):
        # upsample to up_size via nearest then bilinearly resize
        up = resize(bern[i], (up_size, up_size), order=1, mode='reflect',
                    anti_aliasing=False).astype(np.float32)
        x = rng.integers(0, cell)
        y = rng.integers(0, cell)
        masks[i] = up[y:y+img_size, x:x+img_size]
    return torch.from_numpy(masks).to(device)


class RISE:
    """Single-instance RISE saliency."""
    def __init__(self, wrapper: BackboneWrapper, n_masks=N_MASKS, grid=MASK_GRID,
                 p=MASK_P, batch=BATCH, device=None):
        self.wrapper = wrapper
        self.device = device or next(wrapper.model.parameters()).device
        self.n_masks = n_masks
        self.batch = batch
        # Pre-generate masks once (shared across images)
        self.masks = generate_masks(n_masks, grid, p, IMG_SIZE, self.device, seed=42)
        self.mask_sum = self.masks.sum(dim=0)  # for normalisation

    @torch.no_grad()
    def __call__(self, img: torch.Tensor, cidx: int):
        """img: 3xHxW tensor; returns HxW numpy heatmap."""
        x = img.to(self.device).unsqueeze(0)
        scores = torch.empty(self.n_masks, device=self.device)
        for i in range(0, self.n_masks, self.batch):
            j = min(i + self.batch, self.n_masks)
            m = self.masks[i:j].unsqueeze(1)  # B,1,H,W
            masked = x * m
            logits = self.wrapper.model(masked)
            probs = F.softmax(logits, dim=1)
            scores[i:j] = probs[:, cidx]
        # saliency = sum(m_n * s_n) / E[m]
        weighted = (self.masks * scores.view(-1, 1, 1)).sum(dim=0)
        sal = weighted / (self.mask_sum + 1e-8)
        return sal.cpu().numpy()


def per_image_eq(call, img, cidx, angles):
    """Mean Pearson over the requested angles. Returns None if degenerate."""
    try:
        hm0 = call(img, cidx)
    except Exception:
        return None
    if hm0.std() < 1e-8: return 0.0
    vals = []
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
        if np.isfinite(p): vals.append(float(p))
    return float(np.mean(vals)) if vals else None


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('resnet50', 'vgg16'):
        print('usage: rise_baseline.py <resnet50|vgg16>')
        sys.exit(2)
    bb = sys.argv[1]
    out = f'./results_imagenet_official/{bb}__RISE.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[DATA] Loading ImageNet val', flush=True)
    imgs, labs = load_imagenet_val(VAL_DIR, n_per_class=10, max_classes=1000, seed=42)
    print(f'[DATA] {len(imgs)} images; RISE will run on first {max(N_INSDEL, N_EQ)}', flush=True)

    wrapper = BackboneWrapper(bb, device)
    n_use = max(N_INSDEL, N_EQ)
    imgs = imgs[:n_use]; labs = labs[:n_use]
    labs = get_model_predictions(wrapper, imgs, labs)

    rise = RISE(wrapper, n_masks=N_MASKS, batch=BATCH, device=device)
    print(f'[RISE] {N_MASKS} masks ({MASK_GRID}x{MASK_GRID}, p={MASK_P}), batch={BATCH}', flush=True)

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
                'config': {
                    'backbone': bb, 'n_masks': N_MASKS, 'grid': MASK_GRID,
                    'p': MASK_P, 'batch': BATCH, 'n_insdel': N_INSDEL, 'n_eq': N_EQ,
                    'eq_angles': EQ_ANGLES,
                },
                'ins_arr': ins_arr, 'del_arr': del_arr, 'per_image_eq': eq_arr,
                'i_ins_next': i_ins_next, 'i_eq_next': i_eq_next,
            }, f)
        os.replace(tmp, out)

    # Ins/Del pass
    t0 = time.time()
    for i in range(i_ins_next, min(N_INSDEL, len(imgs))):
        try:
            hm = rise(imgs[i], int(labs[i]))
        except Exception as e:
            print(f'  [INS {i}] EXCEPT {e}', flush=True); continue
        ii, dd = insertion_deletion(wrapper, imgs[i], hm, int(labs[i]), 20, 10)
        ins_arr.append(float(ii)); del_arr.append(float(dd))
        i_ins_next = i + 1
        if (i + 1) % 100 == 0:
            save()
            elapsed = (time.time() - t0)
            per = elapsed / (i + 1 - (i_ins_next - len(ins_arr)) + 1e-8)
            remaining = (N_INSDEL - i - 1) * per
            print(f'  [INS {i+1}/{N_INSDEL}] Ins={np.mean(ins_arr):.3f} Del={np.mean(del_arr):.3f}  '
                  f'elapsed={elapsed/3600:.2f}h remaining={remaining/3600:.2f}h', flush=True)

    # Eq pass
    print(f'\n[EQ] starting on n_eq={N_EQ}', flush=True)
    t1 = time.time()
    for i in range(i_eq_next, min(N_EQ, len(imgs))):
        eq = per_image_eq(rise, imgs[i], int(labs[i]), EQ_ANGLES)
        if eq is not None: eq_arr.append(eq)
        i_eq_next = i + 1
        if (i + 1) % 100 == 0:
            save()
            elapsed = (time.time() - t1)
            print(f'  [EQ {i+1}/{N_EQ}] Eq={np.mean(eq_arr):.3f} chunk={elapsed:.0f}s', flush=True)

    # Final result block
    result = {
        'n': len(ins_arr), 'n_eq': len(eq_arr),
        'eq': float(np.mean(eq_arr)) if eq_arr else None,
        'eq_std': float(np.std(eq_arr)) if eq_arr else None,
        'ins': float(np.mean(ins_arr)) if ins_arr else None,
        'ins_std': float(np.std(ins_arr)) if ins_arr else None,
        'del': float(np.mean(del_arr)) if del_arr else None,
        'del_std': float(np.std(del_arr)) if del_arr else None,
    }
    data = {'config': {
        'backbone': bb, 'n_masks': N_MASKS, 'grid': MASK_GRID, 'p': MASK_P,
        'batch': BATCH, 'n_insdel': N_INSDEL, 'n_eq': N_EQ, 'eq_angles': EQ_ANGLES,
    }, 'result': result,
       'ins_arr': ins_arr, 'del_arr': del_arr, 'per_image_eq': eq_arr,
       'i_ins_next': i_ins_next, 'i_eq_next': i_eq_next}
    with open(out, 'w') as f: json.dump(data, f, indent=2, default=str)
    print(f'\n[SAVED] {out}', flush=True)
    print(f'  Eq={result["eq"]:.3f} +/- {result["eq_std"]:.3f}', flush=True)
    print(f'  Ins={result["ins"]:.3f}  Del={result["del"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
