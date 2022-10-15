import torch
import random, os
import numpy as np

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def add_edges(fake_edge_index, fake_edge_weight, training_labels, train_mask, n_real, isolate_nodes=None, threshold=0.9, mode='mix'):
    if isolate_nodes != None:
        node_mask = torch.isin(fake_edge_index[0], isolate_nodes) + torch.isin(fake_edge_index[1], isolate_nodes)
    else:
        node_mask = (fake_edge_index[0] >= n_real) + (fake_edge_index[1] >= n_real)

    if mode == 'mix':
        train_label_mask = torch.logical_and(torch.isin(fake_edge_index[0], train_mask), torch.isin(fake_edge_index[1], train_mask))
        
        sim_mask = torch.logical_and(node_mask, fake_edge_weight >= threshold)

        label_mask = torch.logical_and(node_mask, training_labels[fake_edge_index[0]] == training_labels[fake_edge_index[1]])
        label_mask = torch.logical_and(label_mask, train_label_mask)
        label_mask = torch.logical_and(label_mask, sim_mask)

        sim_mask = torch.logical_and(sim_mask, ~train_label_mask)

        edge_mask = label_mask + sim_mask

    elif mode == 'threshold':
        edge_mask = torch.logical_and(node_mask, fake_edge_weight >= threshold)
    
    else:
        raise AssertionError('Wrong Mode Name')


    fake_edge_index = fake_edge_index[:, edge_mask]
    fake_edge_weight = fake_edge_weight[edge_mask]

    return fake_edge_index, fake_edge_weight, edge_mask

