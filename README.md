<div align="center">

# ArcAD: Anomaly-Rectified Calibration for Cold-Start Supervised Anomaly Detection

[![arXiv](https://img.shields.io/badge/arXiv-2607.02252-b31b1b.svg)](https://arxiv.org/pdf/2607.02252)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#license)

**A plug-and-play calibration framework for reconstruction-based Industrial Anomaly Detection under cold-start conditions.**

</div>

## 🔔 News
- **2026-07**: Code and **data-split JSONs** for MVTec-AD / VisA / Real-IAD / MANTA are released.

## 📖 Introduction

Deploying Industrial Anomaly Detection (IAD) in real manufacturing frequently hits a **cold-start bottleneck**: very few normal samples are available to represent the full normal distribution, and only a handful of anomalies are at hand. Under this regime, existing methods struggle to form a compact normal boundary and fail to exploit the rare supervised defect signal.

**ArcAD** (Anomaly-Rectified Cold-start AD) is a plug-and-play calibration framework built on top of reconstruction-based IAD baselines. Under data scarcity, it constructs a compact and discriminative normal boundary by combining hypersphere-based prototype modeling (**SPM**) with defect-guided contrastive calibration (**DGC**).

Extensive experiments on **MVTec-AD, VisA, Real-IAD, and MANTA** show that ArcAD clearly outperforms state-of-the-art supervised and unsupervised methods in both single-class and multi-class settings under cold-start conditions.

<div align="center">
<img src="figures/framework.png" width="95%">
<br>
<em>Figure 1: Overall framework of ArcAD.</em>
</div>

## 📊 Results

Multi-class unified cold-start setting (all categories trained in a single model).
Each cell reports **I-AUROC / P-AUROC / P-F1-max (%)** (Table 1 of the paper).

| Method | MVTec-AD | VisA | Real-IAD | MANTA |
|--------|----------|------|----------|-------|
| **ArcAD** | 99.7 / 99.2 / 68.9 | 98.9 / 99.0 / 54.9 | 92.5 / 99.0 / 49.8 | 93.3 / 95.5 / 48.5 |

> ArcAD is built upon Dinomaly as the reconstruction baseline. Per-category results are written to `saved_results/<save_name>/results.csv` after evaluation.

## 🛠️ Environment

```bash
conda create -n arcad python=3.8
conda activate arcad
pip install -r requirements.txt
```

Tested on a single NVIDIA RTX 3090 (24GB) with PyTorch 1.12 + CUDA 11.3. Key dependencies: `torch`, `torchvision`, `timm`, `scikit-learn`, `opencv-python-headless`, `tabulate`.

The frozen DINOv2-reg ViT-B/14 backbone weights should be placed at:
```
backbones/weights/dinov2_vitb14_reg4_pretrain.pth
```
(Download from the official DINOv2 release; the registers variant.)

## 📁 Data Preparation

ArcAD adopts a **cold-start supervised** split: a small `labeled` set (few normals + few anomalies, with masks) for training, and a `test` set for evaluation. The exact splits we used are released as open JSON files on 🤗 Hugging Face:

> **Dataset splits:** `<!-- TODO: paste your Hugging Face dataset URL here -->`

Each split JSON is a *manifest* — a list of which images (and their masks) belong to the `labeled` / `test` sets, with paths relative to the dataset root. It does **not** contain the images themselves. To use it, download the original datasets (MVTec-AD, VisA, Real-IAD, MANTA) and arrange them under one root so that the relative paths in the JSON resolve. The released data is organized into cold-start folders named with a `_CD` suffix (`mvtec_CD`, `VisA_CD`, `Real-IAD_CD`, `MANTA_CD`).

### Split JSON format

Every `<category>.json` has the same schema:

```json
{
  "meta":   { "dataset": "mvtec", "category": "bottle", "num_labeled": 69, "num_test": 223 },
  "labeled":[ { "image": "bottle/train/label/good/084.png",  "mask": "",  "label": 0, "anomaly_class": "good" },
              { "image": "bottle/train/label/bad/005.png",   "mask": "bottle/train/label/ground_truth/005_mask.png", "label": 1, "anomaly_class": "defect" } ],
  "test":   [ ... ]
}
```

- All paths are **relative to the dataset root** (the `--data_path` argument).
- `mask` is `""` for normal samples (no mask file).
- `label`: `0` = normal, `1` = anomaly.

The total number of labeled samples matches the cold-start protocol (e.g. MVTec-AD: 1089 normals + 121 anomalies; Real-IAD: 10940 normals + 1216 anomalies).

### Regenerating the splits

If you have organized the raw datasets locally, you can regenerate the split JSONs with [`prepare_data/gen_splits.py`](./prepare_data/gen_splits.py). It enumerates the same `Dataset` classes the training scripts use, so the output is byte-for-byte consistent with the released splits:

```bash
python prepare_data/gen_splits.py --dataset mvtec   --mvtec_root   /path/to/mvtec_CD
python prepare_data/gen_splits.py --dataset visa    --visa_root    /path/to/VisA_CD
python prepare_data/gen_splits.py --dataset realiad --realiad_root /path/to/Real-IAD_CD
python prepare_data/gen_splits.py --dataset manta   --manta_root   /path/to/MANTA_CD
```

### Expected on-disk layout

After obtaining each dataset, arrange it so that the relative paths in the JSON resolve under `--data_path`. Concretely:

<details>
<summary><b>MVTec-AD</b> (mvtec_CD)</summary>

```
<data_path>/bottle/
    train/label/good/*.png
    train/label/bad/*.png
    train/label/ground_truth/<name>_mask.png
    test/good/*.png
    test/<defect_type>/*.png            # e.g. broken_large, contamination, ...
    ground_truth/<defect_type>/<name>_mask.png
```
</details>

<details>
<summary><b>VisA</b> (VisA_CD)</summary>

```
<data_path>/candle/
    train/good/*.JPG
    train/bad/*.JPG
    train/ground_truth/bad/<name>.png
    test/good/*.JPG
    test/bad/*.JPG
    ground_truth/bad/<name>.png
```
</details>

<details>
<summary><b>Real-IAD</b> (Real-IAD_CD)</summary>

```
<data_path>/realiad_1024/<category>/<image>      # image_path from realiad_jsons/sup/<cat>.json
<data_path>/realiad_jsons/sup/<category>.json    # authoritative labeled/test split
```
</details>

<details>
<summary><b>MANTA</b> (MANTA_CD)</summary>

```
<data_path>/MANTA_TINY_256_cropped/<category>/<image>
<data_path>/sup_cropped/<category>.json          # authoritative labeled/test split
```
</details>

## 🚀 Usage

Training and evaluation run in the same script (the model is evaluated on the test set every `eval_freq` iterations and at the end). Each dataset has its own entry-point script.

### Step 1 — Generate prototypes (once per dataset)

Prototypes are K-means centers of the normal-sample bottleneck features, used to initialize SPM. A single generator handles all datasets via `--dataset`:

```bash
python gen_protos.py --dataset mvtec
python gen_protos.py --dataset visa
python gen_protos.py --dataset manta
python gen_protos.py --dataset realiad
```

This writes `prototypes_init.pth` into the dataset's init directory. The number of prototypes `K` defaults to **500** (override with `--num_prototypes N`). Set the GPU with `GEN_DEV=cuda:N` if needed.

> **Single-class setting:** pass `--separate_classes` to emit one prototype file per category (`prototypes_init_<category>.pth`) instead of a single global file.

### Step 2 — Train + evaluate

```bash
# MVTec-AD
python arcad_mvtec_uni.py \
    --data_path /path/to/mvtec_CD \
    --save_name arcad_mvtec

# VisA
python arcad_visa_uni.py \
    --data_path /path/to/VisA_CD \
    --save_name arcad_visa

# Real-IAD
python arcad_realiad_uni.py \
    --data_path /path/to/Real-IAD_CD \
    --save_name arcad_realiad

# MANTA
python arcad_manta_uni.py \
    --data_path /path/to/MANTA_CD \
    --save_name arcad_manta
```

Each script points to the prototype file generated in Step 1 by default; override the path with `--proto_path` if you placed it elsewhere.

Results, checkpoints, and logs are saved under `saved_results/<save_name>/`:
- `results.csv` — per-category + `Mean` metrics
- `model.pth` — final weights
- `log.txt` — training log

## 📂 Repository Structure

```
ArcAD/
├── arcad_{mvtec,visa,manta,realiad}_uni.py   # per-dataset train+eval entry points
├── gen_protos.py                              # unified prototype generator (--dataset)
├── dataset.py                                 # dataset classes (cold-start splits)
├── prepare_data/gen_splits.py                 # export split JSONs (regenerate locally)
├── models/                                    # ViTill reconstruction backbone, SPM, DGC
├── backbones/                                 # frozen DINOv2 encoder loader
├── optimizers/                                # StableAdamW, cosine scheduler
├── utils.py                                   # metrics, logging
└── requirements.txt
```

## 🙏 Acknowledgements

This implementation is built upon [Dinomaly](https://github.com/guojiajeremy/dinomaly) (CVPR 2025), whose minimal reconstruction framework and DINOv2 integration we reuse. We thank the Dinomaly authors. We also thank the maintainers of MVTec-AD, VisA, Real-IAD, and MANTA.

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@article{han2026arcad,
  title   = {ArcAD: Anomaly-Rectified Calibration for Cold-Start Supervised Anomaly Detection},
  author  = {Han, Ningning and Fan, Lei and Guo, Jia and Cao, Yunkang and Su, Xiu and Cao, Feng and Di, Donglin and Su, Tonghua},
  journal = {arXiv preprint arXiv:2607.02252},
  year    = {2026}
}
```

Paper: <https://arxiv.org/pdf/2607.02252>

## License

This project is released under the MIT License.
