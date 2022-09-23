## Transductive Setting

### a) Introduction
The code for the transductive setting semi-supervised learning. 
Including the CORA/CiteSeer/PubMed/Photo/Computers experiment datasets as shown in paper. 
It is implemented mainly based on pytorch_geometric project: https://github.com/rusty1s/pytorch_geometric

### b) Quick Start
- Prepare conda enviroment; more package info can be found in "requirements.txt"
- Set the operations in 'opt.py'; some important operations are listed:
    1. Experiment Dataset (the dataset will be downloaded automatically at the first running time):
       set data_name = ['cora','citeseer','pubmed','photo','computers']
    2. Backbone GNN':\
       set model = ['sgc','ppnp','gcn','gat','sage','cheb']
    3. Training Loss:\
       set loss-name = ['ce','focal','re-weight','cb-softmax']
    4. ReNode Method:\
       set renode-reweight = 1/0 to open/close ReNode\
       set rn-base-weight as the lowerbound of the ReNode Factor\
       set rn-scale-weight as the scale range of the ReNode Factor
    5. Imbalance Issue:\
       set size-imb-type = 'none' if study TINL-only\
       set size-imb-type = 'step' if study TINL&QINL    
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
