# TCRNet-1 Code and Files

This repository contains the code and files for the TCRNet-1 model. For detailed instructions, refer to the PDF in the repository.

TCRNet-1 detects small vehicular targets in infrared imagery by maximizing the target-to-clutter ratio, improving detection accuracy in challenging cluttered environments with limited target pixels.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Generating Chips](#generating-chips)
- [Training TCRNet-1](#training-tcrnet-1)
- [Running and Evaluating TCRNet-1](#running-and-evaluating-tcrnet-1)

## Overview
TCRNet-1 is designed to enhance the detection of small vehicular targets in infrared imagery. It aims to maximize the target-to-clutter ratio, which is essential for accurate detection in environments where the targets are surrounded by clutter.

## Setup
To set up the environment for TCRNet-1, follow these steps:

1. **Install Anaconda**:
   ```bash
   wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
   bash Anaconda3-2020.07-Linux-x86_64.sh
   ```

2. **Create and Activate Conda Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate tcr
   ```

## Generating Chips
Ensure that your data directory has `GT` and `IMAGES` subdirectories. For the ATR Database, `atrdb_gt` contains the appropriate `GT` directory.

### Directory Structure
```
/path/to/processed/data
├── GT
│   ├── cegr01923_0001.txt
│   ├── cegr01923_0002.txt
│   └── ...
└── IMAGES
    ├── cegr01923_0001_0001.png
    ├── cegr01923_0001_0002.png
    ├── ...
    ├── cegr01923_0002_0001.png
    └── cegr01923_0002_0002.png
```

### Running Chip Generation
In Python:
```python
import ds_build
ds_build.build_ds('seqlists/trainlistv3.txt', 'data/exp1', datapath='/path/to/processed/data', skip=30)
```
Or via command line:
```bash
python ds_build.py --seqlist seqlists/trainlistv3.txt --outpath data/exp1 --datapath /path/to/processed/data --skip 30
```

## Training TCRNet-1
Run the following command to train TCRNet-1:
```bash
python train_tcr.py --chippath ./data/exp1 --weightpath ./weights_filters/exp1
```

## Running and Evaluating TCRNet-1
To run and evaluate the model, use the following command:
```bash
python validate_tcr.py --seqlist data/trainlist.txt --weightpath ./weights_filters/exp1 --datapath /path/to/processed/data --skip 30
```
