# AortaSeg

> A block-based pipeline for aortic lumen segmentation, domain inversion, and test-time adaptation.

AortaSeg is a compact research codebase for aortic CTA segmentation. The repository is now organized into clear folders so the training, preprocessing, domain inversion, and TTA workflows are easier to browse on GitHub and easier to maintain locally.

## Highlights

- Clean repository layout with `scripts/` for runnable entry points and `aortaseg/` for reusable library code
- Source-domain training pipeline for 3D block-based segmentation
- Aorta VOI cropping and block generation with TotalSegmentator fallback handling
- Source-like block generation for domain inversion experiments
- Test-time adaptation pipeline with lumen prior refinement and paired metric reporting

## Repository Layout

```text
AortaSeg/
├── aortaseg/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluation.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── utils.py
├── scripts/
│   ├── VOIandBlock_TS.py
│   ├── domain_inversion.py
│   ├── train_sourceCTA_block.py
│   └── ttaCTA_t.py
├── README.md
└── requirements.txt
```

## Workflow

```text
Raw CTA
  ↓
VOI Crop + Block Generation
  ↓
Source Training
  ↓
Domain Inversion
  ↓
Test-Time Adaptation
  ↓
Final AortaSeg Prediction
```

## Main Entry Points

| Script | Purpose |
| --- | --- |
| `scripts/VOIandBlock_TS.py` | Crop the aorta region and export block-level `.npz` files |
| `scripts/train_sourceCTA_block.py` | Train the source-domain AortaSeg model |
| `scripts/domain_inversion.py` | Generate source-like blocks for domain inversion |
| `scripts/ttaCTA_t.py` | Run test-time adaptation on target-domain blocks |

## Core Package

| Module | Responsibility |
| --- | --- |
| `aortaseg/model.py` | 3D AortaSeg network definition |
| `aortaseg/data.py` | Block dataset loading and dataloader creation |
| `aortaseg/losses.py` | Segmentation loss definitions |
| `aortaseg/metrics.py` | Dice, IoU, sensitivity, PPV, HD95, ASSD |
| `aortaseg/evaluation.py` | Case-level reconstruction and evaluation |
| `aortaseg/utils.py` | GPU wait utility and post-processing helpers |

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Optional dependencies:

- `totalsegmentator` is used by `scripts/VOIandBlock_TS.py`
- `pandas` and `openpyxl` are only needed when reading spacing information from Excel

## Quick Start

### 1. Build Aortic Blocks

```bash
python scripts/VOIandBlock_TS.py \
  --data_dir ./data/raw \
  --output_dir ./data/blocks
```

### 2. Train AortaSeg

```bash
python scripts/train_sourceCTA_block.py \
  --data_dir ./data/blocks \
  --checkpoint_dir ./artifacts/AortaSeg
```

### 3. Generate Source-Like Blocks

```bash
python scripts/domain_inversion.py \
  --blocks_dir ./data/blocks \
  --checkpoint ./artifacts/AortaSeg/aortaseg_best.pth \
  --output_blocks_dir ./data/source_like_blocks
```

### 4. Run Test-Time Adaptation

```bash
python scripts/ttaCTA_t.py \
  --checkpoint ./artifacts/AortaSeg/aortaseg_best.pth \
  --s_data_dir ./data/source_like_blocks \
  --t_data_dir ./data/blocks
```

## Notes

- Run all commands from the repository root.
- The `scripts/` entry points automatically resolve the project root, so `python scripts/...` works directly.
- Large outputs such as checkpoints, NIfTI predictions, plots, and CSV artifacts are excluded from Git tracking through `.gitignore`.

## Status

This repository is intentionally kept minimal: the current version focuses on the core AortaSeg pipeline and removes unrelated historical experiments and output artifacts from version control.
