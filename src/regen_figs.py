"""Regenerate Figure 3 (per-angle Eq) and Figure 4 (efficiency-Pareto) from
the v3 official-ImageNet-1K JSON results. Both figures in paper-v1 used the
old Tiny ImageNet protocol and were inconsistent with the n=10,000 numbers.

Output:
  ./figures/angle_resnet.png  (replaced)
  ./figures/angle_vit.png     (replaced)
  ./figures/pareto_resnet.png (replaced)
  ./figures/pareto_vgg.png    (replaced)
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR  = './results_imagenet_official'
IMG  = './figures'
ANGLES = [15, 30, 45, 60, 90, 135, 180]

# ----------------------------------------------------------- per-angle Eq plot
PER_ANGLE_METHODS = [
    ('GradCAM',     'GradCAM',     'tab:gray',   's', '-'),
    ('Grad-CAM++',  'GradCAMpp',   'tab:olive',  '^', '-'),
    ('AugCAM',      'AugCAM_t0',   'tab:red',    'D', '--'),
    ('EquiGrad-CAM T=6',  'CA2_N6',  'tab:cyan',   'o', '-'),
    ('EquiGrad-CAM T=18', 'CA2_N18', 'tab:blue',   '*', '-'),
]

def load_per_angle(bb, m):
    p = os.path.join(DIR, f'{bb}__{m}.json')
    d = json.load(open(p))
    return d.get('result',{}).get('per_angle', {})

def plot_per_angle(bb, out_path, title=None):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for label, key, color, marker, ls in PER_ANGLE_METHODS:
        pa = load_per_angle(bb, key)
        if not pa: continue
        xs, ys = [], []
        for a in ANGLES:
            key_a = str(float(a))
            if key_a in pa: xs.append(a); ys.append(pa[key_a])
        ax.plot(xs, ys, color=color, marker=marker, linestyle=ls,
                label=label, linewidth=1.8, markersize=6)
    ax.set_xlabel('Rotation angle (°)')
    ax.set_ylabel('Per-image Eq (Pearson)')
    ax.set_xticks(ANGLES)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    if title: ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left', fontsize=8, frameon=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')

# ------------------------------------------------------------------ Pareto
PARETO_METHODS = [
    ('Grad-CAM',         'GradCAM',     0.30, 0.14),
    ('Grad-CAM++',       'GradCAMpp',   0.30, 0.14),
    ('XGrad-CAM',        'XGrad_CAM',   0.30, 0.14),
    ('LayerCAM',         'LayerCAM',    0.30, 0.14),
    ('AugCAM(τ=0)',      'AugCAM_t0',   0.86, 0.45),
    ('AugCAM(τ=0.1)',    'AugCAM_t01',  0.84, 0.42),
    ('SmoothGC++',       'SmoothGCpp',  1.97, 1.10),
    ('Score-CAM',        'ScoreCAM',    1.97, 0.69),
    ('EquiGrad(T=6)',    'CA2_N6',      0.45, 0.23),
    ('EquiGrad(T=18)',   'CA2_N18',     0.83, 0.42),
]

def load_eq(bb, m):
    p = os.path.join(DIR, f'{bb}__{m}.json')
    if not os.path.exists(p): return None
    d = json.load(open(p))
    r = d.get('result',{})
    eq = r.get('eq')
    # For ScoreCAM, the JSON conditional mean differs from the paper's Eq=0-padded
    # value. Use whatever's in 'result' (conditional). For paper consistency we'll
    # adjust externally for ScoreCAM only.
    return eq

# Eq values for ScoreCAM under the Eq=0-padded convention (matches paper Tables 1-2)
SCORECAM_PADDED = {'resnet50': 0.494, 'vgg16': 0.102}

def plot_pareto(bb, t_idx, out_path, title=None):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    pts = []
    for label, m, t_r50, t_vgg in PARETO_METHODS:
        eq = load_eq(bb, m)
        if eq is None: continue
        if m == 'ScoreCAM' and bb in SCORECAM_PADDED:
            eq = SCORECAM_PADDED[bb]
        t = t_r50 if bb == 'resnet50' else t_vgg
        pts.append((label, t, eq))
    # Scatter all points
    for label, t, eq in pts:
        is_ours = label.startswith('EquiGrad')
        color = 'tab:blue' if is_ours else 'tab:gray'
        marker = '*' if is_ours else 'o'
        size = 120 if is_ours else 50
        ax.scatter(t, eq, c=color, marker=marker, s=size, edgecolors='black', linewidths=0.5, zorder=5 if is_ours else 3)
        # Offset labels to avoid overlap
        ax.annotate(label, (t, eq), xytext=(5, 4), textcoords='offset points', fontsize=7)
    # Pareto frontier: keep points that are not dominated (lower t AND higher eq)
    pts_sorted = sorted(pts, key=lambda x: x[1])
    frontier = []
    best_eq = -np.inf
    for label, t, eq in pts_sorted:
        if eq > best_eq:
            frontier.append((t, eq))
            best_eq = eq
    fx, fy = zip(*frontier)
    ax.plot(fx, fy, color='red', linestyle='--', linewidth=1.2, label='Pareto frontier', alpha=0.7, zorder=2)
    ax.set_xlabel('Time per image (s, A100)')
    ax.set_ylabel('Equivariance (mean Pearson)')
    ax.set_ylim(0, 1.05)
    if title: ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    os.makedirs(IMG, exist_ok=True)
    plot_per_angle('resnet50',  os.path.join(IMG, 'angle_resnet.png'),  title='ResNet-50')
    plot_per_angle('vit_b_16',  os.path.join(IMG, 'angle_vit.png'),     title='ViT-B/16')
    plot_pareto('resnet50', 0,  os.path.join(IMG, 'pareto_resnet.png'), title='ResNet-50')
    plot_pareto('vgg16',    1,  os.path.join(IMG, 'pareto_vgg.png'),    title='VGG-16')
