# Vision-to-Chart Buoy Association with Learned World-to-Image Projection 
This repository extends the [MaCVi @ CVPR 2026 Vision-to-Chart baseline](https://github.com/mkaraaslan-dev/CVPR2026-Transformer) with a learned world-to-image projection (QueryMLP) that achieves **Overall = 0.7386** (F1 = 0.8055, mIoU = 0.6718) on the held-out test set.

> **Technical report:** *Improved Vision-to-Chart Buoy Association with Learned World-to-Image Projection* — arXiv link pending  
> **Author:** Borja Carrillo-Perez  
> **Challenge:** MaCVi @ CVPR 2026 Vision-to-Chart Data Association Challenge  
> **Challenge overview / results paper:** https://arxiv.org/abs/2604.13244  
> **Baseline architecture:** Kreis and Kiefer, *Real-Time Fusion of Visual and Chart Data for Enhanced Maritime Vision*, arXiv:2507.13880

---

## Method overview

The baseline DETR decoder receives per-buoy queries encoding only world-space distance and bearing, forcing the transformer to learn the full geometric projection implicitly. This solution adds a small frozen MLP (**QueryMLP**) trained to explicitly predict the buoy's waterline contact point in the image from chart measurements and IMU orientation. The predicted pixel coordinates are appended to each query vector, giving the decoder a direct spatial prior and reducing what it must learn from scratch.

![Architecture diagram](assets/diagram.png)

At inference a logit bias of **−0.5** is applied before thresholding (calibrated on the validation set via `sweep_bias_mlp.py`).

### Main changes from the MaCVi baseline

This repository is a fork of the MaCVi Vision-to-Chart Transformer baseline. The main modifications are:

- `query_mlp.py`: defines the frozen QueryMLP that predicts normalized waterline contact points.
- `train_query_mlp.py`: trains QueryMLP from chart, IMU, and annotation correspondences.
- `datasets/buoy_dataset.py`: augments each chart query with QueryMLP-predicted pixel coordinates.
- `training.py`: fine-tunes DETR with 4D queries `[distance, bearing, cx, cy]`.
- `sweep_bias_mlp.py`: calibrates the inference logit bias on the validation split.
- `evaluate.py`: submitted evaluation entry point using the frozen QueryMLP and final DETR checkpoint.

The original 2D-query baseline path is preserved in `training_baseline.py`, `sweep_bias_baseline.py`, `evaluate_baseline.py`, and `datasets/buoy_dataset_baseline.py`.

---

## Results

| Model (split) | P | R | F1 | mIoU | Overall |
|---|---|---|---|---|---|
| Re-trained baseline (val) | 0.7970 | 0.7912 | 0.7941 | 0.6445 | 0.7193 |
| Ours (val) | 0.8627 | 0.7761 | 0.8171 | 0.6753 | **0.7462** |
| Ours (test, leaderboard 2nd place) | 0.8563 | 0.7604 | 0.8055 | 0.6718 | **0.7386** |

The re-trained baseline uses the architecture of Kreis and Kiefer but is trained under the same local conditions as our model: same COCO initialization, hyperparameters, augmentations, and number of epochs. It uses only the original 2D query, distance + bearing, without IMU input or QueryMLP pixel-coordinate prediction.

![Qualitative comparison](assets/comparison_00079.png)

---

## Repository structure

```
macvi26-visionmap-querymlp/
├── query_mlp.py                  ← QueryMLP architecture
├── query_mlp.pth                 ← frozen QueryMLP weights (committed)
├── train_query_mlp.py            ← Step 1: train the frozen MLP
├── training.py                   ← Step 2: fine-tune DETR with 4D queries
├── sweep_bias_mlp.py             ← Step 3: find optimal logit bias on val set
├── evaluate.py                   ← Step 4: run evaluation (submit this)
├── training_baseline.py          ← train the 2D-query baseline for comparison
├── sweep_bias_baseline.py        ← find optimal logit bias for baseline
├── evaluate_baseline.py          ← evaluate the baseline model
├── assets/
│   ├── diagram.png               ← architecture diagram
│   └── comparison_00079.png      ← qualitative comparison figure
├── datasets/
│   ├── buoy_dataset.py           ← modified: integrates frozen QueryMLP
│   └── buoy_dataset_baseline.py  ← original dataset (no QueryMLP)
├── models/                       ← unchanged from baseline
├── util/                         ← unchanged from baseline
├── dataset.yaml                  ← set your dataset paths here
└── Dockerfile                    ← reproducible evaluation environment
```

**Pre-trained / released weights:**

- `query_mlp.pth` — frozen QueryMLP weights, committed to this repository and also included in release `v1.0`.
- `checkpoints/best.pth` — final fine-tuned QueryMLP+DETR-R50 checkpoint, available from GitHub release `v1.0`; required for evaluation.
- `detr-r50-e632da11.pth` — COCO-pretrained DETR-R50 initialization; required only for re-training, not for evaluating the released checkpoint.

---

## Quickstart (Docker)

### 1. Download the submitted DETR checkpoint

```bash
mkdir -p checkpoints
curl -L -o checkpoints/best.pth \
  https://github.com/bcarrpe/macvi26-visionmap-querymlp/releases/download/v1.0/best.pth

`query_mlp.pth` is already in the repo — no download needed
```

### 2. Build the image

```bash
docker build -t buoy-eval .
```

### 3. Run evaluation

By default, `evaluate.py` runs on the public validation split specified in `dataset.yaml`.  
The private test result in the table was obtained from the challenge organizers' held-out evaluation server using the submitted `get_model()` and `input_collate_fn()` implementation.

```bash
docker run --gpus all \
    -v $(pwd):/workspace \
    -v /path/to/dataset-parent:/dataset \
    -it buoy-eval bash -c "cd /workspace && mkdir -p test_results && python evaluate.py"
```

The default `dataset.yaml` assumes the following layout inside the container:

```text
/dataset/aton-dataset/
├── train/
│   ├── images/
│   ├── labels/
│   ├── queries/
│   └── imu/
├── val/
│   ├── images/
│   ├── labels/
│   ├── queries/
│   └── imu/
└── test/
    ├── images/
    ├── labels/
    ├── queries/
    └── imu/
```

Replace `/path/to/aton-dataset` with the local path to the ATON dataset (request access via the [challenge page](https://macvi.org/workshop/cvpr/challenges/vision_map))

Results are printed to stdout and saved to `results.json` and `test_results/np_arr.npy`.

---

## Full training pipeline

All steps below assume you are inside the Docker container (`cd /workspace`).

### Step 1 — Train QueryMLP

```bash
python train_query_mlp.py
```

Outputs `query_mlp.pth`. Trains up to 1000 epochs with early stopping (patience = 60); converges around epoch 585. Expected validation pixel error: median ≈ 18 px.

### Step 2 — Download COCO-pretrained DETR-R50 weights

```bash
wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
```

### Step 3 — Fine-tune DETR

```bash
python training.py
```

Fine-tunes up to 200 epochs (StepLR drop at epoch 135). Best checkpoint saved to `checkpoints/best.pth` (expected around epoch 182).

### Step 4 — Calibrate logit bias

```bash
python sweep_bias_mlp.py
```

Sweeps bias values over [−3, 3] on the validation set. Expected optimal bias: **−0.5**.

### Step 5 — Evaluate

```bash
mkdir -p test_results
python evaluate.py
```

---

## Requirements

- Docker with NVIDIA GPU support (`nvidia-container-toolkit`)
- NVIDIA GPU, CUDA 12.1+, ≥ 8 GB VRAM
- Dataset: ATON dataset (request access via the [challenge page](https://macvi.org/workshop/cvpr/challenges/vision_map))

---

## Citation

If you use this work please cite:

```bibtex
@article{carrilloperez2025v2c,
  author  = {Carrillo-Perez, Borja},
  title   = {Improved Vision-to-Chart Buoy Association with Learned World-to-Image Projection},
  year    = {2026},
  note   = {Technical report},
  note    = {arXiv link TBD}
}
```

And the challenge baseline:

```bibtex
@misc{kreis2025realtime,
  author       = {Kreis, M. and Kiefer, B.},
  title        = {Real-Time Fusion of Visual and Chart Data for Enhanced Maritime Vision},
  year         = {2025},
  eprint       = {2507.13880},
  archivePrefix= {arXiv}
}
```
