import random
import codecs
import copy
import math
import os,sys

import scipy.io as sio

import torch
import torch.nn.functional as F
import numpy as np
from utils import index2dense
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from scipy.sparse import coo_matrix

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

#access a quantity-balanced training set: each class has the same training size train_each
def get_split(opt,all_idx,all_label,nclass = 15):
    train_each = opt.train_each
    valid_each = opt.valid_each

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []
    
    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < train_each:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==train_each*nclass:break

    assert sum(train_list)==train_each*nclass
    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < valid_each:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==valid_each*nclass:break

    assert sum(valid_list)==valid_each*nclass
    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx,valid_idx,test_idx,train_node

#access a quantity-imbalanced training set; the training set follows the step distribution.
def get_step_split(opt,all_idx,all_label,nclass=15):

    base_valid_each = opt.valid_each

    imb_ratio = opt.imb_ratio
    head_list = opt.head_list if len(opt.head_list)>0 else [i for i in range(nclass//2)]

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))

    h_num = len(head_list)
    t_num = len(tail_list)

    base_train_each = int( len(all_idx) * opt.labeling_ratio / (t_num + h_num * imb_ratio) )

    idx2train,idx2valid = {},{}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list: 
        idx2train[i_h] = int(base_train_each * imb_ratio)
        idx2valid[i_h] = int(base_valid_each * 1) 

        total_train_size += idx2train[i_h]
        total_valid_size += idx2valid[i_h]

    for i_t in tail_list: 
        idx2train[i_t] = int(base_train_each * 1)
        idx2valid[i_t] = int(base_valid_each * 1)

        total_train_size += idx2train[i_t]
        total_valid_size += idx2valid[i_t]

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []

    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==total_train_size:break

    assert sum(train_list)==total_train_size

    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==total_valid_size:break

    assert sum(valid_list)==total_valid_size
    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx,valid_idx,test_idx,train_node


#return the ReNode Weight
def get_renode_weight(opt,data_Pi,data_gpr,train_mask,label_matrix):

    ppr_matrix = data_Pi  #personlized pagerank
    gpr_matrix = torch.tensor(data_gpr).float() #class-accumulated personlized pagerank

    base_w  = opt.rn_base_weight
    scale_w = opt.rn_scale_weight
    nnode = ppr_matrix.size(0)
    #train_mask = torch.Tensor(train_mask)
    unlabel_mask = train_mask.int().ne(1)#unlabled node


    #computing the Totoro values for labeled nodes
    gpr_sum = torch.sum(gpr_matrix,dim=1)
    gpr_rn  = gpr_sum.unsqueeze(1) - gpr_matrix
    rn_matrix = torch.mm(ppr_matrix,gpr_rn)

    label_matrix[unlabel_mask] = 0

    rn_matrix = torch.sum(rn_matrix * label_matrix,dim=1)
    rn_matrix[unlabel_mask] = rn_matrix.max() + 99 #exclude the influence of unlabeled node
    
    #computing the ReNode Weight
    train_size    = torch.sum(train_mask.int()).item()
    totoro_list   = rn_matrix.tolist()
    id2totoro     = {i:totoro_list[i] for i in range(len(totoro_list))}
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=False)
    id2rank       = {sorted_totoro[i][0]:i for i in range(nnode)}
    totoro_rank   = [id2rank[i] for i in range(nnode)]
    
    rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(train_size-1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
    rn_weight = rn_weight * train_mask.float()
   
    return rn_weight


#loading the processed data
def load_processed_data(opt,data_path,data_name,shuffle_seed = 0, ppr_file=''):
    

    train_each = opt.train_each
    valid_each = opt.valid_each

    path = "../data/"
    datasetstr = "scence"
    path = path + datasetstr + "/"
    features = sio.loadmat(path + "feat")
    features = features['feat']
    adj = sio.loadmat(path + "adj")
    adj = adj['adj']
    labels = sio.loadmat(path + "label")
    labels = labels['label']
    # if labels.min()==1:
    #     labels = labels-1



    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)


    target_data_num_classes = 15

    all_label = labels.argmax(axis=1)
    target_data_num_nodes = len(all_label)
    all_idx = [i for i in range(len(all_label))]

#    random.seed(shuffle_seed)
    random.shuffle(all_idx)

    train_list = [0 for _ in range(target_data_num_classes)]
    train_node = [[] for _ in range(target_data_num_classes)]
    train_idx = []
    for iter1 in all_idx:
        iter_label = all_label[iter1]
#        if train_list[iter_label] < 50:
        train_list[iter_label] += 1
        train_node[iter_label].append(iter1)
        train_idx.append(iter1)

        if sum(train_list) == train_each*target_data_num_classes: break

    assert sum(train_list) == train_each*target_data_num_classes
    after_train_idx = list(set(all_idx) - set(train_idx))



    valid_idx = after_train_idx[0:500:1]
    test_idx = after_train_idx[500:3735:1]



    target_data_train_mask = torch.zeros(target_data_num_nodes, dtype=torch.bool)
    target_data_valid_mask = torch.zeros(target_data_num_nodes, dtype=torch.bool)
    target_data_test_mask = torch.zeros(target_data_num_nodes, dtype=torch.bool)

    target_data_train_mask_list = test_idx

    target_data_train_mask[torch.tensor(train_idx).long()] = True
    target_data_valid_mask[torch.tensor(valid_idx).long()] = True
    target_data_test_mask[torch.tensor(test_idx).long()] = True



    # calculating the Personalized PageRank Matrix if not exists.
    if os.path.exists(ppr_file):
        target_data_Pi = torch.load(ppr_file)
    else: 
        pr_prob = 1 - opt.pagerank_prob


        adj = adj.toarray()
        A = torch.Tensor(adj)
        A_hat   = A.to(opt.device) + torch.eye(A.size(0)).to(opt.device) # add self-loop
        D       = torch.diag(torch.sum(A_hat,1))
        D       = D.inverse().sqrt()
        A_hat   = torch.mm(torch.mm(D, A_hat), D)
        target_data_Pi = pr_prob * ((torch.eye(A.size(0)).to(opt.device) - (1 - pr_prob) * A_hat).inverse())
        target_data_Pi = target_data_Pi.cpu()
        torch.save(target_data_Pi,ppr_file)

    # calculating the ReNode Weight
    gpr_matrix = [] # the class-level influence distribution
    for iter_c in range(target_data_num_classes):
        iter_Pi = target_data_Pi[torch.tensor(train_node[iter_c]).long()]
        iter_gpr = torch.mean(iter_Pi,dim=0).squeeze()
        gpr_matrix.append(iter_gpr)

    temp_gpr = torch.stack(gpr_matrix,dim=0)
    temp_gpr = temp_gpr.transpose(0,1)
        
    rn_weight =  get_renode_weight(opt,target_data_Pi,temp_gpr,target_data_train_mask,labels) #ReNode Weight

    rn_weight1 = rn_weight.numpy()
    rn_weight12 = rn_weight1.transpose()
    target_data_Pi1 = target_data_Pi.numpy()
    vec = np.dot(target_data_Pi1, rn_weight12)

    vec0 = np.tile(vec, (target_data_num_nodes, 1))
    vec0T = vec0.transpose()

    aij = vec0 * vec0T

#    sio.savemat("mnist10rwnp15w5", {'aij': aij})
    sio.savemat("scence9rnp15w5s30", {'rn_weight1': rn_weight1})
    sio.savemat("scence9trid", {'train_idx': train_idx})
    sio.savemat("scence9vaid", {'valid_idx': valid_idx})
    sio.savemat("scence9teid", {'test_idx': test_idx})

    return rn_weight


