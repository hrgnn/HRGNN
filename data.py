import numpy as np
from numpy.testing import assert_array_almost_equal

from scipy.sparse import csr_matrix

import torch
from deeprobust.graph.utils import preprocess
from deeprobust.graph.data import Dataset, PrePtbDataset

from new_data import New_Dataset
from sklearn.model_selection import train_test_split


def load_data(args, device):
    if args.dataset in ['wisconsin', 'cornell', 'texas', 'film']:
        data = New_Dataset(root='./data/clean_data', name=args.dataset, setting='prognn')
    else:
        data = Dataset(root='./data/clean_data', name=args.dataset, setting='prognn')

    adj, features, labels = preprocess(data.adj, data.features, data.labels, 
                                       preprocess_feature=False, device=device)
        
    nfeat = features.shape[1]
    nclass = labels.max().item()+1
    
    # label rate
    if args.dataset in ['wisconsin', 'cornell', 'texas', 'film']:
        idx_train, idx_test = train_test_split(np.arange(features.shape[0]), test_size=0.8, random_state=args.seed)
        idx_train, idx_val = train_test_split(idx_train, train_size=0.5, random_state=args.seed)
    elif args.dataset == 'pubmed':
        idx_train, idx_val, idx_test = resplit_data(data.idx_train, data.idx_val, data.idx_test, data.labels, ratio=0.001)
    elif args.noise_type == 'label':
        idx_train, idx_val, idx_test = resplit_data(data.idx_train, data.idx_val, data.idx_test, data.labels, ratio=0.03)
    else:
        idx_train, idx_val, idx_test = resplit_data(data.idx_train, data.idx_val, data.idx_test, data.labels, ratio=0.01)

    return features, adj, labels, idx_train, idx_val, idx_test, nfeat, nclass

def load_perturbed_data(args, features, adj, labels, idx_train, idx_val, idx_test, device):
    if args.noise_type == 'metattack':

        perturbed_data = PrePtbDataset(root='./data/perturbed_data/',
                                       name=args.dataset,
                                       attack_method=args.noise_type,
                                       ptb_rate=args.ptb_rate)

        modified_adj = torch.FloatTensor(perturbed_data.adj.todense()).to(device)
        noise_labels = labels

    elif args.noise_type == 'noisy_labels':
        modified_adj = adj
        noise_labels = noisify_labels(labels.cpu().numpy(), idx_train, idx_val, args.noisy_label_rate)
        noise_labels = torch.LongTensor(noise_labels).to(device)

    elif args.noise_type == 'edge_sparsity':
        if type(adj) == torch.Tensor:
            modified_adj = torch.triu(adj)
            edge_index = modified_adj.nonzero().T
        else:
            modified_adj = np.triu(adj.A)
            edge_index = np.array(modified_adj.nonzero())
            
        n_edges = edge_index.shape[1]
        perm = torch.randperm(n_edges)
        num_mask_edges = int(n_edges * args.edge_rate)
        modified_adj[tuple(edge_index[:, perm[num_mask_edges:]])] = 0
        modified_adj = modified_adj + modified_adj.T
        if type(adj) != torch.Tensor:
            row, col = modified_adj.nonzero()
            modified_adj = csr_matrix((modified_adj[row,col], (row, col)), shape=modified_adj.shape)
        
        noise_labels = labels
        
    return modified_adj, noise_labels


def resplit_data(idx_train, idx_val, idx_test, labels, ratio=0.01):
    n_nodes = idx_train.shape[0] + idx_val.shape[0] + idx_test.shape[0]
    n_train_old = idx_train.shape[0]
    n_train_new = int(n_nodes * ratio)
    train_ratio = n_train_new / n_train_old

    classes, counts = np.unique(labels[idx_train], return_counts=True)

    idx_train = idx_train.copy()
    idx_val = idx_val.copy()

    idx_train_new = []
    for c, count in zip(classes, counts):
        n_samples = round(count * train_ratio)
        if n_samples < 1:
            n_samples = 1

        new_idx = idx_train[labels[idx_train]==c]
        np.random.shuffle(new_idx)
        new_train_idx = new_idx[:n_samples]
        new_val_idx = new_idx[n_samples:]

        idx_train_new.append(new_train_idx)
        idx_val = np.concatenate([idx_val, new_val_idx])
        
    idx_train_new = np.concatenate(idx_train_new)
    
    return idx_train_new, idx_val, idx_test

def build_uniform_P(size, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (#class - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = np.float64(noise) / np.float64(size - 1) * np.ones((size, size))
    np.fill_diagonal(P, (np.float64(1)-np.float64(noise))*np.ones(size))
    
    diag_idx = np.arange(size)
    P[diag_idx,diag_idx] = P[diag_idx,diag_idx] + 1.0 - P.sum(0)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def build_pair_p(size, noise):
    assert(noise >= 0.) and (noise <= 1.)
    P = (1.0 - np.float64(noise)) * np.eye(size)
    for i in range(size):
        P[i,i-1] = np.float64(noise)
    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y

def noisify_with_P(y_train, nb_classes, noise, random_state=None, noise_type='uniform'):

    if noise > 0.0:
        if noise_type=='uniform':
            print('Uniform noise')
            P = build_uniform_P(nb_classes, noise)
        elif noise_type == 'pair':
            print('Pair noise')
            P = build_pair_p(nb_classes, noise)
        else:
            print('Noise type have implemented')
        # seed the random numbers with #run
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    else:
        P = np.eye(nb_classes)

    return torch.LongTensor(y_train), P
    
def noisify_labels(labels, idx_train, idx_val, noise):
    n_class = labels.max().item()+1

    y_train_noise = noisify_with_P(labels[idx_train], n_class, noise)[0]
    y_val_noise = noisify_with_P(labels[idx_val], n_class, noise)[0]
    
    labels[idx_train] = y_train_noise
    labels[idx_val] = y_val_noise

    return labels




    
