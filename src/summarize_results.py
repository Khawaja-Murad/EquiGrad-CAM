"""Aggregate the three per-backbone JSON outputs into a single table."""
import json
import os
from pathlib import Path

RESULTS = Path.home() / "scratch/ca2gradcam/results_imagenet"
BACKBONES = ["resnet50", "vgg16", "vit_b_16"]
METHODS = ["GradCAM","GradCAM++","XGrad-CAM","LayerCAM","SmoothGC++",
           "ScoreCAM","AugCAM_t0","AugCAM_t01","CA2_N6","CA2_N18"]

print(f"\n{'='*100}")
print(f"  CA2 Grad-CAM — ImageNet-1K aggregate")
print(f"{'='*100}\n")

for bb in BACKBONES:
    fp = RESULTS / f"ImageNet1K_{bb}.json"
    if not fp.exists():
        print(f"\n[{bb}] MISSING: {fp}")
        continue
    d = json.load(open(fp))
    cfg = d.get("config", {})
    print(f"\n[{bb}]  n={cfg.get('n_total','?')}  n_eq={cfg.get('n_eq','?')}")
    print(f"{'Method':<16} {'N':>5} {'Eq':>7} {'±std':>7} {'Ins':>7} {'Del':>7} {'t/img':>7}")
    print(f"{'-'*70}")
    for m in METHODS:
        r = d.get(m, {})
        if "eq" not in r: continue
        print(f"{m:<16} {r.get('n','?'):>5} {r['eq']:>7.3f} {r['eq_std']:>7.3f} "
              f"{r['ins']:>7.3f} {r['del']:>7.3f} {r['time_img']:>6.2f}s")

    print(f"\n  Significance vs CA2_N18:")
    for m, st in d.get("significance", {}).items():
        if "p" not in st: continue
        sig = "***" if st['p']<0.001 else "**" if st['p']<0.01 else "*" if st['p']<0.05 else "n.s."
        print(f"    {m:<16} p={st['p']:.2e} {sig:<4} Δeq={st.get('diff',0):+.3f}")

    print(f"\n  Ablation:")
    for m, r in d.get("ablation", {}).items():
        print(f"    {m:<16} Eq={r['eq']:.3f} Ins={r['ins']:.3f}")

print(f"\n{'='*100}")
print("Files:")
for bb in BACKBONES:
    fp = RESULTS / f"ImageNet1K_{bb}.json"
    if fp.exists():
        sz = fp.stat().st_size
        print(f"  {fp.name}  ({sz/1024:.1f} KB)")
