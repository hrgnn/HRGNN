# HRGNN

A PyTorch implementation of "HRGNN: Learning Holistic Robustness Graph Neural Networks on Noisy Graphs with Label Scarcity"

The code is based on our Pytorch adversarial repository, DeepRobust [(https://github.com/DSE-MSU/DeepRobust)](https://github.com/DSE-MSU/DeepRobust)


## Requirements
```
numpy
torch
scipy
deeprobust
scikit_learn
torch_geometric
```

## Installation
To run the code, first you need to install DeepRobust:
```
pip install deeprobust
```
Or you can clone it and install from source code:
```
git clone https://github.com/DSE-MSU/DeepRobust.git
cd DeepRobust
python setup.py install
```

## Run the code
After installation, you can clone this repository
```
git clone https://github.com/hrgnn/HRGNN.git
cd HRGNN
python main.py --dataset cora --noise_type metattack --ptb_rate 0.2
```

## Reproduce the results
All the hyper-parameters settings are included in [`scripts`](https://github.com/hrgnn/HRGNN/tree/master/scripts) folder. Note that same hyper-parameters are used under different type of noise for the same dataset. 

To reproduce the performance reported in the paper, you can run the bash files in folder `scripts`.
```
sh scripts/cora.sh
```

<!--
**IMPORTANT NOTICE** For the performance of Pubmed dataset, if you don't add the code (line 59-62 in `train.py`), the performance of GCN should be around 85 since the data splits are different from what I used in the paper. See details in https://github.com/ChandlerBang/Pro-GNN/issues/2
-->