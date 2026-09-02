"""CUB-200-2011 localisation (IoU + Pointing Game) for a chosen backbone.

Identical heatmap→bbox protocol as run_cub_localization.py (threshold-ratio
0.3, largest connected component). Wraps it to take a backbone argument so
we can close the VGG-16 / ViT-B/16 gap flagged by R2.

For ViT-B/16, ScoreCAM is excluded (incompatible with patch tokens, same
convention as ImageNet Tables).

Invocation:
  python run_cub_localization_bb.py <backbone>
where backbone ∈ {resnet50, vgg16, vit_b_16}.

Output: results_cub/CUB200_localization_<backbone>.json
"""
import os, sys, json, time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, '.')
from ca2_complete_eval import (
    set_seed, BackboneWrapper,
    GradCAM_v2, GradCAMPP, XGradCAM, LayerCAM, ScoreCAM,
    SmoothGradCAMPP, AugCAM_v2, CA2GradCAM_v2,
    _make_caller,
)
from scipy.ndimage import label as cc_label

CUB_ROOT = './CUB_200_2011'


def load_cub_test_with_bboxes(max_images=1000, seed=42):
    images_txt   = open(os.path.join(CUB_ROOT, 'images.txt')).read().splitlines()
    bbox_txt     = open(os.path.join(CUB_ROOT, 'bounding_boxes.txt')).read().splitlines()
    labels_txt   = open(os.path.join(CUB_ROOT, 'image_class_labels.txt')).read().splitlines()
    split_txt    = open(os.path.join(CUB_ROOT, 'train_test_split.txt')).read().splitlines()
    id_to_path  = {int(l.split()[0]): l.split()[1] for l in images_txt}
    id_to_bbox  = {int(l.split()[0]): tuple(map(float, l.split()[1:])) for l in bbox_txt}
    id_to_class = {int(l.split()[0]): int(l.split()[1]) - 1 for l in labels_txt}
    id_to_test  = {int(l.split()[0]): (l.split()[1] == '0') for l in split_txt}
    test_ids = sorted([i for i, is_test in id_to_test.items() if is_test])
    rng = np.random.default_rng(seed)
    rng.shuffle(test_ids)
    test_ids = test_ids[:max_images]
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    pre = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    data = []
    for img_id in test_ids:
        path = os.path.join(CUB_ROOT, 'images', id_to_path[img_id])
        try:
            im = Image.open(path).convert('RGB')
        except Exception:
            continue
        W0, H0 = im.size
        x, y, w, h = id_to_bbox[img_id]
        scale = 256.0 / min(W0, H0)
        x1, y1, x2, y2 = x*scale, y*scale, (x+w)*scale, (y+h)*scale
        W1, H1 = W0*scale, H0*scale
        cx, cy = (W1 - 224)/2, (H1 - 224)/2
        bx1, by1 = max(0, x1-cx), max(0, y1-cy)
        bx2, by2 = min(224, x2-cx), min(224, y2-cy)
        if bx2 <= bx1 or by2 <= by1:
            continue
        tensor = pre(im)
        data.append({
            'image_id': img_id,
            'path': id_to_path[img_id],
            'tensor': tensor,
            'bbox_224': (float(bx1), float(by1), float(bx2), float(by2)),
            'class': id_to_class[img_id],
        })
    return data


def heatmap_to_bbox(hm, threshold_ratio=0.3):
    mask = hm > (hm.max() * threshold_ratio)
    if not mask.any():
        return None
    labeled, n = cc_label(mask)
    if n == 0: return None
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    comp = (labeled == largest)
    ys, xs = np.where(comp)
    return (float(xs.min()), float(ys.min()), float(xs.max()+1), float(ys.max()+1))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    aarea = max(0, (ax2-ax1)) * max(0, (ay2-ay1))
    barea = max(0, (bx2-bx1)) * max(0, (by2-by1))
    u = aarea + barea - inter
    return 0.0 if u <= 0 else inter / u


def pointing_hit(hm, bbox_224):
    y, x = np.unravel_index(int(np.argmax(hm)), hm.shape)
    x1, y1, x2, y2 = bbox_224
    return float(x1 <= x < x2 and y1 <= y < y2)


def main():
    if len(sys.argv) < 2:
        print('usage: run_cub_localization_bb.py <backbone>')
        sys.exit(2)
    bb = sys.argv[1]
    assert bb in ('resnet50', 'vgg16', 'vit_b_16')
    out_path = f'./results_cub/CUB200_localization_{bb}.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[DATA] Loading CUB-200-2011 test set with bboxes...', flush=True)
    data = load_cub_test_with_bboxes(max_images=1000, seed=42)
    print(f'[DATA] {len(data)} usable test images for {bb}', flush=True)

    wrapper = BackboneWrapper(bb, device)

    methods = {
        'GradCAM':    GradCAM_v2(wrapper),
        'GradCAM++':  GradCAMPP(wrapper),
        'XGrad-CAM':  XGradCAM(wrapper),
        'LayerCAM':   LayerCAM(wrapper),
        'SmoothGC++': SmoothGradCAMPP(wrapper, n_samples=50),
        'AugCAM_t0':  AugCAM_v2(wrapper, tau=0.0),
        'AugCAM_t01': AugCAM_v2(wrapper, tau=0.10),
        'CA2_N6':     CA2GradCAM_v2(wrapper, tau=0.10, n_angles=6),
        'CA2_N18':    CA2GradCAM_v2(wrapper, tau=0.10, n_angles=18),
    }
    if bb != 'vit_b_16':
        methods['ScoreCAM'] = ScoreCAM(wrapper, batch_size=64)

    labs = []
    for d in data:
        pred, _ = wrapper.predict(d['tensor'])
        labs.append(pred)

    results = {}
    if os.path.exists(out_path):
        try: results = json.load(open(out_path))
        except: results = {}

    def save():
        tmp = out_path + '.tmp'
        with open(tmp, 'w') as f: json.dump(results, f, indent=2, default=str)
        os.replace(tmp, out_path)

    for name, m in methods.items():
        if name in results and 'iou_mean' in (results[name] or {}):
            print(f'  [{name}] SKIP', flush=True); continue
        call = _make_caller(m)
        ious, hits = [], []
        t0 = time.time()
        for i, (d, cidx) in enumerate(zip(data, labs)):
            try:
                hm = call(d['tensor'], cidx)
            except Exception as e:
                continue
            pred_bbox = heatmap_to_bbox(hm, threshold_ratio=0.3)
            if pred_bbox is None: ious.append(0.0)
            else: ious.append(iou(pred_bbox, d['bbox_224']))
            hits.append(pointing_hit(hm, d['bbox_224']))
            if (i+1) % 200 == 0:
                print(f'  [{name}] {i+1}/{len(data)}  IoU={np.mean(ious):.3f}  PG={np.mean(hits):.3f}', flush=True)
        results[name] = {
            'n':         len(ious),
            'iou_mean':  float(np.mean(ious)),
            'iou_std':   float(np.std(ious)),
            'pg':        float(np.mean(hits)),
            'time_img':  (time.time()-t0)/max(1, len(ious)),
        }
        save()
        print(f'  [{name}] DONE  IoU={results[name]["iou_mean"]:.3f}±{results[name]["iou_std"]:.3f}  PG={results[name]["pg"]:.3f}', flush=True)
    print(f'[SAVED] {out_path}', flush=True)


if __name__ == '__main__':
    main()
