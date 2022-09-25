## ReNode_weight matrix

## Introduction
We use part of the code NeurIPS 2021 paper "[Topology-Imbalance Learning for Semi-Supervised Node Classification](https://arxiv.org/abs/2110.04099)", to generate the ReNode_weight matrix with the Scene15 datasets as example. The ReNode_weight matrix can be used in [ReNode-GLCNMR](https://github.com/BiJingjun/ReNode-GLCNMR). 
One can change the datasets' information in 'load_data.py' to generate the ReNode_weight matrix with other datasets.


## Quick Start
- Prepare conda enviroment; more package info can be found in "requirements.txt"
- Set the operations in 'opt.py'
- Running command: 'python transductive_run.py'
      

## Citation
```
@inproceedings{chen2021renode,
  author    = {Deli, Chen and Yankai, Lin and Guangxiang, Zhao and Xuancheng, Ren and Peng, Li and Jie, Zhou and Xu, Sun},
  title     = {{Topology-Imbalance Learning for Semi-Supervised Node Classification}},
  booktitle = {NeurIPS},
  year      = {2021}
}
```
