#!/usr/bin/env python3
"""
EquiGrad-CAM — Complete Self-Contained Evaluation
==================================================
Single file. No external dependencies beyond torch/torchvision/scipy/numpy/PIL.
Includes ALL methods, ALL ablations, ImageNet-1K + Tiny ImageNet support.

USAGE ON your compute environment:
  # 1. Set IMAGENET_VAL_DIR below
  # 2. Run:
  python ca2_complete_eval.py

  # Or import and run specific experiments:
  from ca2_complete_eval import *
  results = run_imagenet_eval("resnet50", "/data/imagenet/val", ...)
"""

import os, sys, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from scipy.stats import wilcoxon
from PIL import Image


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_CFG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "tau": 0.10,
    "n_angles": 18,
    "pad": 64,
    "n_steps": 20,
    "blur_sigma": 10,
    "eq_angles": [15.0, 30.0, 45.0, 60.0, 90.0, 135.0, 180.0],
    "eq_n": 500,
    "seed": 42,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD  = torch.tensor([0.229, 0.224, 0.225])

def to_rgb(t):
    x = t.cpu().clone()
    for i in range(3): x[i] = x[i] * STD[i] + MEAN[i]
    return x.permute(1, 2, 0).clamp(0, 1).numpy()

def rpbh_rotate(img, angle, pad=64):
    """Rotate with reflective padding boundary handling."""
    C, H, W = img.shape
    padded = F.pad(img.unsqueeze(0), [pad]*4, mode='reflect').squeeze(0)
    pH, pW = padded.shape[1], padded.shape[2]
    theta = torch.tensor([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
        [np.sin(np.radians(angle)),  np.cos(np.radians(angle)), 0]
    ], dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, (1, C, pH, pW), align_corners=False)
    rotated = F.grid_sample(padded.unsqueeze(0), grid, mode='bilinear',
                             padding_mode='zeros', align_corners=False).squeeze(0)
    cy, cx = (pH - H) // 2, (pW - W) // 2
    return rotated[:, cy:cy+H, cx:cx+W]

def inv_rotate(tensor, angle):
    """Inverse-rotate a feature tensor (C, H, W) by -angle."""
    if abs(angle) < 1e-6: return tensor
    C, h, w = tensor.shape
    theta = torch.tensor([
        [np.cos(np.radians(-angle)), -np.sin(np.radians(-angle)), 0],
        [np.sin(np.radians(-angle)),  np.cos(np.radians(-angle)), 0]
    ], dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, (1, C, h, w), align_corners=False)
    return F.grid_sample(tensor.unsqueeze(0), grid, mode='bilinear',
                          padding_mode='zeros', align_corners=False).squeeze(0)

def rotate_heatmap_np(hm, angle):
    """Rotate a numpy heatmap."""
    t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0).float()
    theta = torch.tensor([
        [np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
        [np.sin(np.radians(angle)),  np.cos(np.radians(angle)), 0]
    ], dtype=torch.float32).unsqueeze(0)
    grid = F.affine_grid(theta, t.shape, align_corners=False)
    out = F.grid_sample(t, grid, mode='bilinear', padding_mode='zeros',
                         align_corners=False)
    return out.squeeze().numpy()

def _make_caller(method):
    """Closure-safe caller factory."""
    m = method
    def call(img, cidx):
        r = m(img, cidx)
        return r[0] if isinstance(r, tuple) else r
    return call


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BACKBONE WRAPPER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BackboneWrapper:
    CONFIGS = {
        "resnet50": ("layer4", (7, 7), False),
        "vgg16":    ("features", (14, 14), False),
        "vit_b_16": ("encoder", (14, 14), True),
    }

    def __init__(self, name, device="cuda"):
        self.device = device
        self.name = name
        self.is_vit = self.CONFIGS[name][2]
        self.spatial_size = self.CONFIGS[name][1]
        self._acts, self._grads = [], []

        if name == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V1").to(device).eval()
            layer = self.model.layer4[-1]
        elif name == "vgg16":
            self.model = models.vgg16(weights="IMAGENET1K_V1").to(device).eval()
            # register_full_backward_hook conflicts with inplace ReLU (view+inplace
            # autograd safeguard), so flip every inplace ReLU off before hooking.
            for _m in self.model.modules():
                if isinstance(_m, nn.ReLU):
                    _m.inplace = False
            layer = self.model.features[29]
        elif name == "vit_b_16":
            self.model = models.vit_b_16(weights="IMAGENET1K_V1").to(device).eval()
            layer = self.model.encoder.layers[-1].ln_1
        else:
            raise ValueError(f"Unknown backbone: {name}")

        self._fwd = layer.register_forward_hook(self._hook_fwd)
        self._bwd = layer.register_full_backward_hook(self._hook_bwd)

    def _hook_fwd(self, m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        self._acts = [o.detach()]

    def _hook_bwd(self, m, gi, go):
        g = go[0] if isinstance(go, tuple) else go
        self._grads = [g.detach()]

    def gradcam_pass(self, x, cidx):
        self.model.zero_grad()
        x = x.to(self.device)
        logits = self.model(x)
        score = logits[0, cidx]
        score.backward()
        acts = self._acts[0][0].cpu()
        grads = self._grads[0][0].cpu()
        if self.is_vit:
            if acts.shape[0] == 197: acts = acts[1:]
            if grads.shape[0] == 197: grads = grads[1:]
            D = acts.shape[1]
            h = w = int(acts.shape[0] ** 0.5)
            acts = acts.permute(1, 0).reshape(D, h, w)
            grads = grads.permute(1, 0).reshape(D, h, w)
        probs = torch.softmax(logits, dim=1)
        conf = float(probs[0, cidx])
        pred = int(probs.argmax(1)[0])
        return acts, grads, conf, pred

    def predict(self, img):
        with torch.no_grad():
            x = img.unsqueeze(0).to(self.device) if img.dim() == 3 else img.to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            return int(probs.argmax(1)[0]), float(probs.max(1)[0])

    def score(self, x, cidx):
        with torch.no_grad():
            if x.dim() == 3: x = x.unsqueeze(0)
            logits = self.model(x.to(self.device))
            return float(torch.softmax(logits, dim=1)[0, cidx])

    def remove_hooks(self):
        self._fwd.remove(); self._bwd.remove()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HEATMAP GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_heatmap_v2(acts, grads, hw=(224, 224), relu_weights=False):
    w = grads.mean(dim=(1, 2))
    if relu_weights: w = F.relu(w)
    cam = (w[:, None, None] * acts).sum(0)
    cam = F.relu(cam)
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), hw,
                         mode="bilinear", align_corners=False).squeeze()
    mn, mx = cam.min(), cam.max()
    if mx > mn: cam = (cam - mn) / (mx - mn)
    return cam


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ALL SALIENCY METHODS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GradCAM_v2:
    def __init__(self, wrapper): self.w = wrapper
    def __call__(self, img, cidx):
        self.w.model.eval()
        A, g, _, _ = self.w.gradcam_pass(img.unsqueeze(0), cidx)
        return make_heatmap_v2(A, g, (img.shape[1], img.shape[2]),
                                relu_weights=self.w.is_vit).numpy()

class GradCAMPP:
    def __init__(self, wrapper): self.w = wrapper
    def __call__(self, img, cidx):
        self.w.model.eval(); self.w.model.zero_grad()
        x = img.unsqueeze(0).to(self.w.device)
        logits = self.w.model(x)
        logits[0, cidx].backward(retain_graph=True)
        grads = self.w._grads[0][0].cpu()
        acts = self.w._acts[0][0].cpu()
        if self.w.is_vit:
            if acts.shape[0] == 197: acts = acts[1:]; grads = grads[1:]
            D = acts.shape[1]; h = int(acts.shape[0]**0.5)
            acts = acts.permute(1,0).reshape(D,h,h)
            grads = grads.permute(1,0).reshape(D,h,h)
        g2 = grads**2; g3 = grads**3
        s = (acts * g3).sum(dim=(1,2), keepdim=True)
        alpha = g2 / (2*g2 + s + 1e-8)
        alpha = F.relu(alpha)
        w = (alpha * F.relu(grads)).sum(dim=(1,2))
        cam = (w[:,None,None] * acts).sum(0)
        cam = F.relu(cam)
        H, W = img.shape[1], img.shape[2]
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (H,W),
                             mode="bilinear", align_corners=False).squeeze()
        mn, mx = cam.min(), cam.max()
        if mx > mn: cam = (cam - mn)/(mx - mn)
        return cam.numpy()

class XGradCAM:
    def __init__(self, wrapper): self.w = wrapper
    def __call__(self, img, cidx):
        self.w.model.eval()
        A, g, _, _ = self.w.gradcam_pass(img.unsqueeze(0), cidx)
        w = (g * A).sum(dim=(1,2)) / (A.sum(dim=(1,2)) + 1e-8)
        cam = (w[:,None,None] * A).sum(0)
        cam = F.relu(cam)
        H, W = img.shape[1], img.shape[2]
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (H,W),
                             mode="bilinear", align_corners=False).squeeze()
        mn, mx = cam.min(), cam.max()
        if mx > mn: cam = (cam - mn)/(mx - mn)
        return cam.numpy()

class LayerCAM:
    def __init__(self, wrapper): self.w = wrapper
    def __call__(self, img, cidx):
        self.w.model.eval()
        A, g, _, _ = self.w.gradcam_pass(img.unsqueeze(0), cidx)
        cam = (F.relu(g) * A).sum(0)
        cam = F.relu(cam)
        H, W = img.shape[1], img.shape[2]
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (H,W),
                             mode="bilinear", align_corners=False).squeeze()
        mn, mx = cam.min(), cam.max()
        if mx > mn: cam = (cam - mn)/(mx - mn)
        return cam.numpy()

class ScoreCAM:
    def __init__(self, wrapper, batch_size=32):
        self.w = wrapper; self.bs = batch_size
    def __call__(self, img, cidx):
        self.w.model.eval()
        H, W = img.shape[1], img.shape[2]
        x = img.unsqueeze(0).to(self.w.device)
        with torch.no_grad(): _ = self.w.model(x)
        acts = self.w._acts[0].detach()
        if acts.dim() == 4:
            acts = acts[0]
        K, h, w = acts.shape
        up = F.interpolate(acts.unsqueeze(0), (H,W), mode='bilinear',
                            align_corners=False).squeeze(0)
        for k in range(K):
            mn, mx = up[k].min(), up[k].max()
            if mx > mn: up[k] = (up[k]-mn)/(mx-mn)
        with torch.no_grad():
            base = F.softmax(self.w.model(x), dim=1)[0, cidx].item()
        scores = torch.zeros(K)
        for s in range(0, K, self.bs):
            e = min(s+self.bs, K)
            masked = x * up[s:e].unsqueeze(1)
            with torch.no_grad():
                out = F.softmax(self.w.model(masked.to(self.w.device)), dim=1)
            scores[s:e] = out[:, cidx].cpu()
        weights = F.relu(scores - base)
        cam = (weights[:,None,None].to(acts.device) * acts).sum(0)
        cam = F.relu(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), (H,W),
                             mode='bilinear', align_corners=False).squeeze()
        mn, mx = cam.min(), cam.max()
        if mx > mn: cam = (cam-mn)/(mx-mn)
        return cam.cpu().numpy()

class SmoothGradCAMPP:
    def __init__(self, wrapper, n_samples=50, noise_std=0.15):
        self.w = wrapper; self.n = n_samples; self.std = noise_std
        self.gcpp = GradCAMPP(wrapper)
    def __call__(self, img, cidx):
        self.w.model.eval()
        hms = [self.gcpp(img + torch.randn_like(img)*self.std, cidx)
               for _ in range(self.n)]
        avg = np.mean(hms, axis=0)
        mn, mx = avg.min(), avg.max()
        if mx > mn: avg = (avg-mn)/(mx-mn)
        return avg

class AugCAM_v2:
    def __init__(self, wrapper, n_angles=18, tau=0.0, pad=64):
        self.w = wrapper
        self.angles = list(np.linspace(-180, 180, n_angles, endpoint=False))
        self.tau = tau; self.pad = pad
    def __call__(self, img, cidx):
        self.w.model.eval()
        H, W = img.shape[1], img.shape[2]
        maps, ws = [], []
        for a in self.angles:
            rot = rpbh_rotate(img, a, self.pad)
            A, g, conf, pred = self.w.gradcam_pass(rot.unsqueeze(0), cidx)
            if self.tau > 0 and (pred != cidx or conf < self.tau): continue
            hm = make_heatmap_v2(A, g, (H,W), relu_weights=self.w.is_vit).numpy()
            maps.append(hm); ws.append(max(conf, 1e-8))
        if not maps: return GradCAM_v2(self.w)(img, cidx)
        ws = np.array(ws)/sum(ws)
        agg = sum(w*m for w,m in zip(ws, maps))
        mn, mx = agg.min(), agg.max()
        return (agg-mn)/(mx-mn+1e-8)

class CA2GradCAM_v2:
    def __init__(self, wrapper, tau=0.10, n_angles=18, use_ggas=False, pad=64):
        self.w = wrapper; self.tau = tau; self.n_angles = n_angles
        self.use_ggas = use_ggas; self.pad = pad
    def __call__(self, img, cidx, return_peum=False):
        self.w.model.eval()
        H, W = img.shape[1], img.shape[2]
        is_vit = self.w.is_vit
        acc_A, acc_g, acc_w, pv_hms = [], [], [], []
        for a in np.linspace(-180, 180, self.n_angles, endpoint=False):
            rot = rpbh_rotate(img, float(a), self.pad)
            A, g, conf, pred = self.w.gradcam_pass(rot.unsqueeze(0), cidx)
            if pred != cidx or conf < self.tau: continue
            tA = inv_rotate(A, a); tg = inv_rotate(g, a)
            acc_A.append(tA); acc_g.append(tg); acc_w.append(conf)
            if return_peum:
                vh = make_heatmap_v2(tA, tg, (H,W), relu_weights=is_vit).numpy()
                pv_hms.append((vh, conf))
        if not acc_A:
            hm = GradCAM_v2(self.w)(img, cidx)
            return (hm, np.zeros_like(hm)) if return_peum else hm
        total = sum(acc_w); ws = [w/total for w in acc_w]
        barA = sum(w*A for w,A in zip(ws, acc_A))
        barg = sum(w*g for w,g in zip(ws, acc_g))
        hm = make_heatmap_v2(barA, barg, (H,W), relu_weights=is_vit).numpy()
        peum = None
        if return_peum and pv_hms:
            pws = [c/total for _,c in pv_hms]
            peum = sum(pw*(vh-hm)**2 for (vh,_),pw in zip(pv_hms, pws))
        return (hm, peum) if return_peum else hm


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ABLATION VARIANTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class iGradCAM:
    def __init__(self, wrapper, n_angles=18, pad=64):
        self.w = wrapper; self.n = n_angles; self.pad = pad
    def __call__(self, img, cidx):
        self.w.model.eval()
        H, W = img.shape[1], img.shape[2]
        aA, ag = [], []
        for a in np.linspace(-180, 180, self.n, endpoint=False):
            rot = rpbh_rotate(img, float(a), self.pad)
            A, g, _, _ = self.w.gradcam_pass(rot.unsqueeze(0), cidx)
            aA.append(inv_rotate(A, a)); ag.append(inv_rotate(g, a))
        bA = torch.stack(aA).mean(0); bg = torch.stack(ag).mean(0)
        return make_heatmap_v2(bA, bg, (H,W), relu_weights=self.w.is_vit).numpy()

class iGradCAM_PCF:
    def __init__(self, wrapper, n_angles=18, tau=0.10, pad=64):
        self.w = wrapper; self.n = n_angles; self.tau = tau; self.pad = pad
    def __call__(self, img, cidx):
        self.w.model.eval()
        H, W = img.shape[1], img.shape[2]
        aA, ag = [], []
        for a in np.linspace(-180, 180, self.n, endpoint=False):
            rot = rpbh_rotate(img, float(a), self.pad)
            A, g, conf, pred = self.w.gradcam_pass(rot.unsqueeze(0), cidx)
            if pred != cidx or conf < self.tau: continue
            aA.append(inv_rotate(A, a)); ag.append(inv_rotate(g, a))
        if not aA: return GradCAM_v2(self.w)(img, cidx)
        bA = torch.stack(aA).mean(0); bg = torch.stack(ag).mean(0)
        return make_heatmap_v2(bA, bg, (H,W), relu_weights=self.w.is_vit).numpy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ViT-SPECIFIC BASELINES
# Attention Rollout (Abnar & Zuidema 2020)
# Transformer Attribution (Chefer et al. CVPR 2021)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ViTAttentionRecorder:
    """Wrap a torchvision vit_b_16 to expose per-layer attention matrices.

    Each encoder block's forward is replaced so that self_attention is called
    with need_weights=True, average_attn_weights=False. The attention tensor
    of shape (B, H, T, T) is retained for backward and stored in self.attns.
    """
    def __init__(self, vit_model, device='cuda'):
        self.model = vit_model.to(device).eval()
        self.device = device
        self.is_vit = True
        self.attns = []
        self._patch_blocks()

    def _patch_blocks(self):
        """Re-implement each EncoderBlock's forward, computing self-attention
        manually so the attention matrix is on the autograd graph.

        nn.MultiheadAttention's fused kernel does not produce a grad-tracked
        attention tensor even with need_weights=True, so we bypass it.
        """
        outer = self
        for blk in self.model.encoder.layers:
            ln_1 = blk.ln_1
            attn = blk.self_attention
            dropout = blk.dropout
            ln_2 = blk.ln_2
            mlp = blk.mlp

            num_heads = attn.num_heads
            embed_dim = attn.embed_dim
            head_dim  = embed_dim // num_heads
            scale     = head_dim ** -0.5
            in_proj_w = attn.in_proj_weight
            in_proj_b = attn.in_proj_bias
            out_proj  = attn.out_proj

            def make_fwd(ln_1=ln_1, dropout=dropout, ln_2=ln_2, mlp=mlp,
                          num_heads=num_heads, head_dim=head_dim, scale=scale,
                          in_proj_w=in_proj_w, in_proj_b=in_proj_b, out_proj=out_proj):
                def _fwd(input):
                    B, T, D = input.shape
                    x = ln_1(input)
                    qkv = F.linear(x, in_proj_w, in_proj_b)              # (B, T, 3D)
                    q, k, v = qkv.chunk(3, dim=-1)
                    def split(t):
                        return t.reshape(B, T, num_heads, head_dim).permute(0, 2, 1, 3)
                    q, k, v = split(q), split(k), split(v)               # (B, H, T, D/H)
                    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
                    A = F.softmax(scores, dim=-1)
                    if A.requires_grad:
                        A.retain_grad()
                    outer.attns.append(A)
                    ctx = torch.matmul(A, v)                              # (B, H, T, D/H)
                    ctx = ctx.permute(0, 2, 1, 3).reshape(B, T, D)
                    x_out = out_proj(ctx)
                    x_out = dropout(x_out)
                    x_out = x_out + input
                    y = ln_2(x_out)
                    y = mlp(y)
                    return x_out + y
                return _fwd
            blk.forward = make_fwd()

    def reset(self):
        self.attns = []

    def predict(self, img):
        with torch.no_grad():
            x = img.unsqueeze(0).to(self.device) if img.dim() == 3 else img.to(self.device)
            self.reset()
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            return int(probs.argmax(1)[0]), float(probs.max())

    def score(self, img, cidx):
        """For insertion/deletion: softmax confidence of class cidx."""
        with torch.no_grad():
            x = img.unsqueeze(0).to(self.device) if img.dim() == 3 else img.to(self.device)
            self.reset()
            logits = self.model(x)
            return float(torch.softmax(logits, dim=1)[0, cidx])


class AttentionRollout:
    """Abnar & Zuidema (2020) — multiplicative rollout over (A + I)/2.

    Per layer i: A_i_bar = (A_i.mean(heads) + I) / 2, row-normalised.
    Rollout R = A_L_bar @ ... @ A_1_bar.
    Heatmap = R[0, 0, 1:] reshaped to (h, w) → upsampled to image size.
    """
    def __init__(self, recorder, head_fusion='mean'):
        self.w = recorder
        self.head_fusion = head_fusion

    def __call__(self, img, cidx):
        H, W = img.shape[1], img.shape[2]
        with torch.no_grad():
            x = img.unsqueeze(0).to(self.w.device)
            self.w.reset()
            _ = self.w.model(x)
        if not self.w.attns:
            return np.zeros((H, W), dtype=np.float32)
        roll = None
        for A in self.w.attns:
            if self.head_fusion == 'mean':
                a = A.mean(dim=1)         # (B, T, T)
            elif self.head_fusion == 'max':
                a, _ = A.max(dim=1)
            else:
                a = A.min(dim=1).values
            I = torch.eye(a.shape[-1], device=a.device).unsqueeze(0)
            a = (a + I) / 2
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            roll = a if roll is None else torch.bmm(a, roll)
        cls_row = roll[0, 0, 1:]
        side = int(round(cls_row.shape[0] ** 0.5))
        cam = cls_row.reshape(1, 1, side, side).float()
        cam = F.interpolate(cam, (H, W), mode='bilinear',
                             align_corners=False).squeeze()
        cam = cam.clamp_min(0)
        mn, mx = cam.min(), cam.max()
        if mx > mn: cam = (cam - mn) / (mx - mn)
        return cam.cpu().numpy()


class TransformerAttribution:
    """Chefer et al. (CVPR 2021) — gradient-weighted attention rollout.

    For each layer i:
        C_bar_i = (∂y_c/∂A_i ⊙ A_i).clamp_min(0).mean(over heads)
    Then R = I; for each layer: R += C_bar_i @ R.
    Heatmap = R[0, 0, 1:] reshaped to (h, w) → upsampled.
    """
    def __init__(self, recorder):
        self.w = recorder

    def __call__(self, img, cidx):
        H, W = img.shape[1], img.shape[2]
        self.w.model.zero_grad()
        self.w.reset()
        x = img.unsqueeze(0).to(self.w.device).requires_grad_(False)
        logits = self.w.model(x)
        score = logits[0, cidx]
        score.backward(retain_graph=False)
        attns = self.w.attns
        if not attns:
            return np.zeros((H, W), dtype=np.float32)
        T = attns[0].shape[-1]
        R = torch.eye(T, device=self.w.device).unsqueeze(0)
        for A in attns:
            G = A.grad
            if G is None:
                continue
            Cbar = (G * A).clamp_min(0).mean(dim=1)   # (B, T, T)
            R = R + torch.bmm(Cbar, R)
        cls_row = R[0, 0, 1:]
        side = int(round(cls_row.shape[0] ** 0.5))
        cam = cls_row.detach().reshape(1, 1, side, side).float()
        cam = F.interpolate(cam, (H, W), mode='bilinear',
                             align_corners=False).squeeze()
        cam = cam.clamp_min(0)
        mn, mx = cam.min(), cam.max()
        if mx > mn: cam = (cam - mn) / (mx - mn)
        return cam.cpu().numpy()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# EVALUATION FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def insertion_deletion(wrapper, img, hm, cidx, n_steps=20, sigma=10):
    H, W = img.shape[1], img.shape[2]
    blur = transforms.GaussianBlur(kernel_size=51, sigma=sigma)
    blurred = blur(img.unsqueeze(0)).squeeze(0)
    order = np.argsort(-hm.flatten())
    n_px = H * W
    step = max(1, n_px // n_steps)
    mask = torch.zeros(1, 1, H, W)
    ins_scores, del_scores = [wrapper.score(blurred, cidx)], [wrapper.score(img, cidx)]
    for i in range(step, n_px+1, step):
        idxs = order[max(0,i-step):i]
        rows, cols = idxs // W, idxs % W
        mask[0, 0, rows, cols] = 1.0
        ins_img = img * mask.squeeze(0) + blurred * (1 - mask.squeeze(0))
        ins_scores.append(wrapper.score(ins_img, cidx))
        del_img = blurred * mask.squeeze(0) + img * (1 - mask.squeeze(0))
        del_scores.append(wrapper.score(del_img, cidx))
    ins_auc = float(np.trapz(ins_scores, dx=1.0/len(ins_scores)))
    del_auc = float(np.trapz(del_scores, dx=1.0/len(del_scores)))
    return ins_auc, del_auc

def eq_scores(method_fn, imgs, cidxs, angles):
    from scipy.stats import pearsonr
    all_p, per_angle = [], {a: [] for a in angles}
    for img, cidx in zip(imgs, cidxs):
        hm0 = method_fn(img, cidx)
        for a in angles:
            rot = rpbh_rotate(img, float(a))
            hm_r = method_fn(rot, cidx)
            ref = rotate_heatmap_np(hm0, float(a))
            if hm_r.std() > 1e-8 and ref.std() > 1e-8:
                p, _ = pearsonr(hm_r.flatten(), ref.flatten())
                all_p.append(float(p)); per_angle[a].append(float(p))
    return {
        "pearson": float(np.mean(all_p)) if all_p else 0.0,
        "pearson_std": float(np.std(all_p)) if all_p else 0.0,
        "ssim": 0.0,
        "per_angle": {a: float(np.mean(v)) if v else 0.0 for a,v in per_angle.items()},
    }

def compute_per_image_eq(method_fn, imgs, cidxs, angles):
    from scipy.stats import pearsonr
    per_img = []
    for img, cidx in zip(imgs, cidxs):
        hm0 = method_fn(img, cidx)
        scores = []
        for a in angles:
            rot = rpbh_rotate(img, float(a))
            hm_r = method_fn(rot, cidx)
            ref = rotate_heatmap_np(hm0, float(a))
            if hm_r.std() > 1e-8 and ref.std() > 1e-8:
                p, _ = pearsonr(hm_r.flatten(), ref.flatten())
                scores.append(float(p))
        per_img.append(float(np.mean(scores)) if scores else float('nan'))
    return per_img

def get_model_predictions(wrapper, imgs, labs):
    new_labs = []
    for img in imgs:
        pred, _ = wrapper.predict(img)
        new_labs.append(pred)
    n_match = sum(1 for a,b in zip(labs, new_labs) if a == b)
    print(f"[LABELS] Model predictions for {len(imgs)} images "
          f"(match dataset: {n_match}/{len(imgs)})")
    confs = []
    for img, cidx in zip(imgs, new_labs):
        _, c = wrapper.predict(img)
        confs.append(c)
    print(f"  Confidence: mean={np.mean(confs):.3f}, "
          f"median={np.median(confs):.3f}, min={np.min(confs):.3f}")
    return new_labs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_tiny(root, split="train", n_per=5, seed=42):
    np.random.seed(seed)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        print(f"[ERROR] {split_dir} not found"); return [], []
    classes = sorted(os.listdir(split_dir))
    classes = [c for c in classes if os.path.isdir(os.path.join(split_dir, c))]
    imgs, labs = [], []
    for ci, cls in enumerate(classes):
        img_dir = os.path.join(split_dir, cls, "images") if split == "train" \
                  else os.path.join(split_dir, cls)
        if not os.path.isdir(img_dir): img_dir = os.path.join(split_dir, cls)
        files = sorted([f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.jpeg','.jpg','.png'))])
        sel = np.random.choice(len(files), min(n_per, len(files)), replace=False)
        for fi in sel:
            try:
                im = Image.open(os.path.join(img_dir, files[fi])).convert('RGB')
                imgs.append(transform(im)); labs.append(ci)
            except: continue
    print(f"[DATA] TinyImageNet ({split}): {len(imgs)} imgs, "
          f"{len(classes)} classes × up to {n_per}")
    return imgs, labs

def load_imagenet_val(val_dir, n_per_class=10, max_classes=500, seed=42):
    np.random.seed(seed)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    subdirs = sorted([d for d in os.listdir(val_dir)
                      if os.path.isdir(os.path.join(val_dir, d))])
    if not subdirs:
        print(f"[ERROR] No class dirs in {val_dir}"); return [], []
    selected = subdirs[:max_classes]
    print(f"[DATA] ImageNet-1K val: {len(subdirs)} classes found, "
          f"using {len(selected)}, {n_per_class}/class")
    imgs, labs = [], []
    for ci, cls in enumerate(selected):
        cls_path = os.path.join(val_dir, cls)
        files = sorted([f for f in os.listdir(cls_path)
                       if f.lower().endswith(('.jpeg','.jpg','.png'))])
        if not files: continue
        sel = np.random.choice(len(files), min(n_per_class, len(files)), replace=False)
        for fi in sel:
            try:
                im = Image.open(os.path.join(cls_path, files[fi])).convert('RGB')
                imgs.append(transform(im)); labs.append(ci)
            except: continue
    print(f"  Loaded {len(imgs)} images")
    return imgs, labs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXPERIMENT RUNNER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_full_eval(backbone, imgs, labs, cfg, out_dir, dataset_name="ImageNet"):
    """Run complete 10-method evaluation."""
    set_seed(cfg.get("seed", 42))
    os.makedirs(out_dir, exist_ok=True)
    device = cfg["device"]

    print(f"\n{'='*70}")
    print(f"  EquiGrad-CAM — {dataset_name} — {backbone}")
    print(f"  {len(imgs)} images")
    print(f"{'='*70}\n")

    wrapper = BackboneWrapper(backbone, device)
    print(f"  Spatial={wrapper.spatial_size}  ViT={wrapper.is_vit}")
    labs = get_model_predictions(wrapper, imgs, labs)

    n_eq = min(cfg["eq_n"], len(imgs))
    eq_angles = cfg["eq_angles"]

    methods = {
        "GradCAM":     GradCAM_v2(wrapper),
        "GradCAM++":   GradCAMPP(wrapper),
        "XGrad-CAM":   XGradCAM(wrapper),
        "LayerCAM":    LayerCAM(wrapper),
        "SmoothGC++":  SmoothGradCAMPP(wrapper, n_samples=50),
        "AugCAM_t0":   AugCAM_v2(wrapper, tau=0.0),
        "AugCAM_t01":  AugCAM_v2(wrapper, tau=0.10),
        "CA2_N6":      CA2GradCAM_v2(wrapper, tau=cfg["tau"], n_angles=6),
        "CA2_N18":     CA2GradCAM_v2(wrapper, tau=cfg["tau"], n_angles=18),
    }
    if not wrapper.is_vit:
        methods["ScoreCAM"] = ScoreCAM(wrapper, batch_size=64)

    out_path = os.path.join(out_dir, f"{dataset_name}_{backbone}.json")

    # Incremental save + resume support
    results = {}
    per_image_eq = {}
    if os.path.exists(out_path):
        try:
            loaded = json.load(open(out_path))
            per_image_eq = loaded.pop("_per_image_eq", {})
            results = loaded
            done = [k for k in results
                    if isinstance(results.get(k), dict) and "eq" in results.get(k, {})]
            print(f"[RESUME] {out_path} exists; skipping methods already done: {done}",
                  flush=True)
        except Exception as e:
            print(f"[RESUME] failed to load existing JSON: {e}", flush=True)
            results, per_image_eq = {}, {}

    def save_partial():
        tmp = out_path + ".tmp"
        data = {**results, "_per_image_eq": per_image_eq}
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, out_path)

    for name, method in methods.items():
        if (name in results and isinstance(results[name], dict)
                and "eq" in results[name]):
            print(f"\n  [{name}] SKIP (already in {out_path})", flush=True)
            continue

        n_use = len(imgs)
        n_eq_use = min(n_eq, n_use)

        print(f"\n  [{name}] ({n_use} imgs, {n_eq_use} eq) ...", flush=True)
        t0 = time.time()
        call = _make_caller(method)

        ins_l, del_l = [], []
        for idx, (img, cidx) in enumerate(zip(imgs[:n_use], labs[:n_use])):
            hm = call(img, cidx)
            i, d = insertion_deletion(wrapper, img, hm, cidx,
                                       cfg["n_steps"], cfg["blur_sigma"])
            ins_l.append(i); del_l.append(d)
            if (idx+1) % 500 == 0:
                print(f"    {idx+1}/{n_use} (Ins={np.mean(ins_l):.3f})", flush=True)

        print(f"    Equivariance ({n_eq_use} imgs)...", flush=True)
        eqs = eq_scores(call, imgs[:n_eq_use], labs[:n_eq_use], eq_angles)
        pie = compute_per_image_eq(call, imgs[:n_eq_use], labs[:n_eq_use], eq_angles)
        per_image_eq[name] = pie

        wall = time.time() - t0
        results[name] = {
            "n": n_use, "n_eq": n_eq_use,
            "eq": eqs["pearson"], "eq_std": eqs["pearson_std"],
            "ins": float(np.mean(ins_l)), "ins_std": float(np.std(ins_l)),
            "del": float(np.mean(del_l)), "del_std": float(np.std(del_l)),
            "time_img": wall/max(n_use,1),
            "per_angle": eqs["per_angle"],
        }
        r = results[name]
        print(f"    Eq={r['eq']:.3f}±{r['eq_std']:.3f} Ins={r['ins']:.3f} "
              f"Del={r['del']:.3f} t={r['time_img']:.2f}s")
        save_partial()
        print(f"    [SAVED partial] {out_path}", flush=True)

    # Significance
    print(f"\n[STATS] vs CA2_N18")
    stats = {}
    ca2 = per_image_eq.get("CA2_N18", [])
    for name in per_image_eq:
        if name == "CA2_N18": continue
        a = per_image_eq[name]; b = ca2[:len(a)]
        valid = [(x,y) for x,y in zip(a,b)
                 if np.isfinite(x) and np.isfinite(y) and x!=y]
        if len(valid) < 10:
            stats[name] = {"p": float("nan")}; continue
        av=[v[0] for v in valid]; bv=[v[1] for v in valid]
        stat, p = wilcoxon(av, bv)
        diff = np.mean(bv)-np.mean(av)
        stats[name] = {"p":float(p),"n":len(valid),"diff":float(diff)}
        sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "n.s."
        print(f"  {name:16s}: p={p:.2e} {sig} Δ={diff:+.3f}")

    results["significance"] = stats
    results["config"] = {"backbone":backbone, "dataset":dataset_name,
                          "n_total":len(imgs), "n_eq":n_eq}
    save_partial()

    # Ablation
    print(f"\n[ABLATION] Component decomposition")
    abl_methods = {
        "GradCAM": GradCAM_v2(wrapper),
        "i-GradCAM": iGradCAM(wrapper, n_angles=18),
        "+PCF": iGradCAM_PCF(wrapper, n_angles=18, tau=cfg["tau"]),
        "CA2_full": CA2GradCAM_v2(wrapper, tau=cfg["tau"], n_angles=18),
    }
    ablation = results.get("ablation", {}) if isinstance(results.get("ablation"), dict) else {}
    for name, method in abl_methods.items():
        if name in ablation and "eq" in ablation[name]:
            print(f"  {name:16s}: SKIP (already done)")
            continue
        call = _make_caller(method)
        eqs = eq_scores(call, imgs[:min(n_eq,100)], labs[:min(n_eq,100)], eq_angles)
        ins_l = []
        for img, cidx in zip(imgs[:200], labs[:200]):
            hm = call(img, cidx)
            i, _ = insertion_deletion(wrapper, img, hm, cidx,
                                       cfg["n_steps"], cfg["blur_sigma"])
            ins_l.append(i)
        ablation[name] = {"eq":eqs["pearson"],"ins":float(np.mean(ins_l))}
        results["ablation"] = ablation
        print(f"  {name:16s}: Eq={ablation[name]['eq']:.3f} Ins={ablation[name]['ins']:.3f}")
        save_partial()

    # Final save (idempotent with save_partial; keep for clarity)
    save_partial()
    print(f"\n[SAVED] {out_path}")

    # Print table
    print(f"\n{'='*85}")
    print(f"COMPLETE TABLE — {backbone} — {dataset_name} — {len(imgs)} images")
    print(f"{'='*85}")
    print(f"{'Method':<16} {'N':>5} {'Eq':>7} {'±std':>7} {'Ins':>7} {'Del':>7} {'t/img':>7}")
    print(f"{'-'*85}")
    for name in ["GradCAM","GradCAM++","XGrad-CAM","LayerCAM","SmoothGC++",
                  "ScoreCAM","AugCAM_t0","AugCAM_t01","CA2_N6","CA2_N18"]:
        if name in results and isinstance(results[name],dict) and "eq" in results[name]:
            r = results[name]
            print(f"{name:<16} {r['n']:>5} {r['eq']:>7.3f} {r['eq_std']:>7.3f} "
                  f"{r['ins']:>7.3f} {r['del']:>7.3f} {r['time_img']:>6.2f}s")

    wrapper.remove_hooks()
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONVENIENCE WRAPPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_imagenet_eval(backbone, val_dir, n_per_class=10, max_classes=500,
                       n_eq=500, device="cuda", out_dir="./results_imagenet"):
    cfg = {**DEFAULT_CFG, "device": device, "eq_n": n_eq}
    imgs, labs = load_imagenet_val(val_dir, n_per_class, max_classes)
    if not imgs: return {}
    return run_full_eval(backbone, imgs, labs, cfg, out_dir, "ImageNet1K")

def run_tinyimagenet_eval(backbone, tiny_root, n_per=5, n_eq=100,
                           device="cuda", out_dir="./results_tiny"):
    cfg = {**DEFAULT_CFG, "device": device, "eq_n": n_eq}
    imgs, labs = load_tiny(tiny_root, "train", n_per)
    if not imgs: return {}
    return run_full_eval(backbone, imgs, labs, cfg, out_dir, "TinyImageNet")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # ===== CONFIGURE THESE =====
    IMAGENET_VAL = "/data/imagenet/val"   # ← SET THIS FOR your compute environment
    TINY_ROOT    = None                    # Set if running Tiny ImageNet too
    BACKBONE     = "resnet50"              # resnet50, vgg16, vit_b_16
    N_PER_CLASS  = 10                      # 10 × 500 = 5000 images
    MAX_CLASSES  = 500                     # 500 or 1000
    N_EQ         = 500                     # equivariance images
    OUT_DIR      = "./ca2_results"
    # ===========================

    print("EquiGrad-CAM Complete Evaluation")
    print(f"Backend: {BACKBONE}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    if os.path.isdir(IMAGENET_VAL):
        run_imagenet_eval(BACKBONE, IMAGENET_VAL,
                          n_per_class=N_PER_CLASS,
                          max_classes=MAX_CLASSES,
                          n_eq=N_EQ,
                          out_dir=OUT_DIR)
    elif TINY_ROOT and os.path.isdir(TINY_ROOT):
        run_tinyimagenet_eval(BACKBONE, TINY_ROOT, out_dir=OUT_DIR)
    else:
        print(f"[ERROR] Neither {IMAGENET_VAL} nor {TINY_ROOT} found.")
        print("Set IMAGENET_VAL or TINY_ROOT at the bottom of this script.")
