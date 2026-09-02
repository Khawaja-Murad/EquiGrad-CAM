# EquiGrad-CAM

Reference implementation and evaluation code for
*"Signal or Noise? Auditing Rotation-Induced Saliency Drift in Medical and Aerial
Imaging"* by Khawaja Murad ul Hassan (Independent Researcher) and Mehran Ebrahimi
(Ontario Tech University).

**EquiGrad-CAM** is the method; the paper asks whether the rotation-induced drift in
Grad-CAM is faithful signal or an artefact of the CAM operator, and removes it where it
is the latter.

EquiGrad-CAM aggregates rotated views of an input **in feature space, before
upsampling**, yielding rotation-equivariant Grad-CAM-style explanations. It defines a
one-parameter family of operating points: the equivariance-optimal `τ=0` endpoint
(**EquiGrad-CAM**) and the prediction-conditioned `τ=0.10` variant (**EquiGrad-CAM+PCF**).

## Layout

```
src/
  ca2_complete_eval.py          # core: saliency methods, hooks, BackboneWrapper, metrics
                                #   (EquiGrad-CAM = class iGradCAM (τ=0) / CA2GradCAM_v2(τ=0.10) = +PCF)
  rise_baseline.py              # RISE
  opticam_baseline.py           # Opti-CAM (2024)
  reciprocam_baseline.py        # Recipro-CAM (2024)
  run_official_eq10k.py         # main ImageNet-1K equivariance / Ins / Del (n_eq=10,000)
  run_ablation_n1000.py         # component-decomposition ablation
  run_augscorecam_v2.py         # Augmented Score-CAM
  equal_compute_runner.py       # equal-wall-clock RISE-400 / Score-CAM-80
  run_peum_validation.py        # PEUM image-level / per-pixel calibration
  run_cub_localization_bb.py    # CUB-200-2011 IoU / Pointing Game per backbone
  run_vit_baselines_eq10k.py    # Attention Rollout / Transformer Attribution (ViT)
  run_tau_sweep_resnet50.py     # PCF threshold (τ) sweep
  run_agg_strategy_resnet50.py  # aggregation-formula ablation
  run_vit_hook_ablation.py      # ViT hook-location ablation
  run_ssim_ablation.py          # SSIM equivariance metric
  regen_figs.py                 # angle / Pareto figures
  --- operator analysis and hardening ---
  exp_operator_decomposition.py # stage-wise Eq of the CAM operator (acts/grads/alpha/...)
  exp_causal_drift.py           # causal occlusion test: drift vs agreement vs random region
  exp_r2_conditional_eq.py      # Eq conditioned on the model's own prediction stability
  analyze_conditional_eq.py     #   -> summary table over all backbones (CPU)
  run_T_sweep.py                # Eq as a function of the number of aggregated views T
  exp_peum_audit.py             # PEUM raw per-image records (held-out / bootstrap ready)
  analyze_peum_audit.py         #   -> lift, bootstrap CIs, baselines, AUROC (CPU)
  run_rise_vit.py               # RISE on ViT-B/16
  download_imagenet_official_v2.py, download_cub.py   # dataset setup
```

## Method ↔ code mapping

| Paper                | Code |
|----------------------|------|
| EquiGrad-CAM (τ=0)   | `iGradCAM` in `ca2_complete_eval.py` |
| EquiGrad-CAM+PCF (τ=0.10) | `CA2GradCAM_v2(tau=0.10)` |
| Grad-CAM / ++ / XGrad / LayerCAM / Score-CAM / Smooth GC++ / AugCAM | classes in `ca2_complete_eval.py` |

(The class prefix `CA2`/`iGradCAM` is the internal development name for EquiGrad-CAM.)

## Setup

```bash
pip install -r requirements.txt
# Point the runners at your ImageNet-1K val and CUB-200-2011 directories
# (see the top of each runner / ca2_complete_eval.py).
python src/download_imagenet_official_v2.py     # optional: pull the non-gated HF mirror
python src/download_cub.py                       # optional: CUB-200-2011
```

Experiments use the **full official ILSVRC-2012 ImageNet-1K validation set**
(evaluation does `Resize(256)+CenterCrop(224)`). 10/class × 1000 classes,
deterministic `seed=42`. `download_imagenet_official_v2.py` is a convenience fetcher;
point the runners at your own copy of the official validation set if you have one.

**Sampling matters.** The loader is class-blocked (10 consecutive images per class), so a
*prefix* of length k covers only k/10 classes and is an easier sample: on ResNet-50 the
first 591 images inflate Eq by +0.07. Reduced-n runs here are class-spread, never
prefixes. See `SAMPLING` notes at the top of each runner. The target class is the model's top-1 on the unrotated image,
held fixed across rotations.

## Reproducing the main results

```bash
python src/run_official_eq10k.py resnet50      # ImageNet-1K Eq / Ins / Del, all baselines
python src/run_official_eq10k.py vgg16
python src/run_official_eq10k.py vit_b_16
python src/run_cub_localization_bb.py resnet50 # CUB IoU / Pointing Game
python src/run_peum_validation.py              # PEUM calibration (r=0.561 image-level)
python src/equal_compute_runner.py             # matched-wall-clock baselines
```

### Operator analysis (where the drift enters, and whether it is faithful)

```bash
python src/exp_operator_decomposition.py --backbone resnet50 --n_eq 2000
python src/exp_causal_drift.py           --backbone resnet50 --n_eq 1000 --topk 0.10
python src/exp_r2_conditional_eq.py      --backbone resnet50 --start 0 --count 200
python src/analyze_conditional_eq.py                       # CPU, summarises all backbones
python src/run_T_sweep.py                --backbone resnet50 --n_eq 1000
python src/exp_peum_audit.py             --backbone resnet50 --n_eq 2000
python src/analyze_peum_audit.py         --backbone resnet50 --n_boot 10000   # CPU
```

`exp_operator_decomposition.py` reports equivariance at every stage of the Grad-CAM
chain. Its final stage reproduces the deployed single-view Grad-CAM Eq to within 0.003,
which is the check that the decomposition matches the real operator. Note that on
ResNet-50 the gradient stage is reported as undefined: a GAP+linear head makes
`d(y_c)/d(A_kij)` independent of `(i,j)`, so every gradient channel is spatially constant
and a spatial correlation does not exist. That is a structural property, not a failure.

Metrics: equivariance is the Pearson correlation between the heatmap of the rotated
image and the rotated heatmap, averaged over 7 evaluation angles
{15,30,45,60,90,135,180}°; insertion/deletion AUC use a 20-step Gaussian-blur baseline
(σ=10); CUB localisation thresholds the heatmap at 0.3×max, takes the largest connected
component's box, and reports IoU + Pointing Game.

## License

Released under the MIT License (see `LICENSE`).
