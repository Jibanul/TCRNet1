
# TCRNet-1 code and files

See also the readme pdf from UCF.

## Setup
Get conda installed, e.g.,
```
wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh
```

Make conda env
```
conda env create -f environment.yml
conda activate tcr
```

## Generate chips

This code is assuming that there is a data directory with GT and IMAGES subdirectories.  For the ATR Database, atrdb_gt contains the appropriate "GT" directory.  We  also assume that the IMAGES directory has exported png files.  Something like this:

- /path/to/processed/data
  - GT
    - cegr01923_0001.txt
    - cegr01923_0002.txt
    - ...
  - IMAGES
    - cegr01923_0001_0001.png
    - cegr01923_0001_0002.png
    - ...
    - cegr01923_0002_0001.png
    - cegr01923_0002_0002.png
    - ...

### Running chip generation
(in python)
```
>>> import ds_build
>>> ds_build.build_ds('seqlists/trainlistv3.txt', 'data/exp1', datapath='/path/to/processed/data', skip=30)
```
OR
```
python ds_build.py --seqlist seqlists/trainlistv3.txt --outpath data/exp1 --datapath /path/to/processed/data --skip 30
```
## Optimize filters 
Note: this is no longer necessary as a separate step; it's incorporated into train_tcr
```
python qcf_basis.py --chippath ./data/exp1 --weightpath ./weights_filters/exp1
```
## train tcrnet
```
python train_tcr.py --chippath ./data/exp1 --weightpath ./weights_filters/exp1
```
## run and evaluate tcrnet
```
python validate_tcr.py --seqlist data/trainlist.txt --weightpath ./weights_filters/exp1 --datapath /path/to/processed/data --skip 30
```
