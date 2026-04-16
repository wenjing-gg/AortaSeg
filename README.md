# AortaSeg

AortaSeg is a minimal block-based pipeline for aortic lumen segmentation.

Main entry points:

- `VOIandBlock_TS.py`: crop the aorta region with TotalSegmentator and build 3D blocks
- `train_sourceCTA_block.py`: train the source-domain AortaSeg model
- `domain_inversion.py`: generate source-like blocks for domain inversion
- `ttaCTA_t.py`: run test-time adaptation on target-domain blocks

Install:

```bash
pip install -r requirements.txt
```

Example workflow:

```bash
python VOIandBlock_TS.py --data_dir ./data/raw --output_dir ./data/blocks
python train_sourceCTA_block.py --data_dir ./data/blocks --checkpoint_dir ./artifacts/AortaSeg
python domain_inversion.py --blocks_dir ./data/blocks --checkpoint ./artifacts/AortaSeg/aortaseg_best.pth --output_blocks_dir ./data/source_like_blocks
python ttaCTA_t.py --checkpoint ./artifacts/AortaSeg/aortaseg_best.pth --s_data_dir ./data/source_like_blocks --t_data_dir ./data/blocks
```
