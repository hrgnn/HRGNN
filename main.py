import argparse
import numpy as np
import torch

from model import HRGNN
from utils import seed_everything
from data import load_data, load_perturbed_data

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0, help='Gpu id')
parser.add_argument('--seed', type=int, default=15, help='Random seed')
parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'cora_ml', 'pubmed', 'wisconsin', 'cornell', 'texas', 'film'])

parser.add_argument('--noise_type', type=str, default='edge_sparsity', choices=['clean', 'metattack', 'edge_sparsity', 'noisy_labels'])
parser.add_argument('--edge_rate', type=float, default=1.0, choices=[0.0, 0.2, 0.5, 0.8, 1.0], help='Edge rate')
parser.add_argument('--ptb_rate', type=float, default=0.2, choices=[0.0, 0.05, 0.1, 0.15, 0.2], help='Pertubation rate')
parser.add_argument('--noisy_label_rate', type=float, default=0.2, help='Flipped label rate')

# model training setting
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train on HRGNN in a iteration.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')

parser.add_argument('--k', type=int, default=3)
parser.add_argument('--iteration', '-i', type=int, default=10)
parser.add_argument('--fake_nodes', '-f', type=int, default=20)
parser.add_argument('--mask_rate', '-m', type=float, default=0.2)
parser.add_argument('--threshold', '-t', type=float, default=0.9)
parser.add_argument('--add_labels', '-add', type=int, default=250)
args = parser.parse_args()

def main(args):
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    
    if args.ptb_rate == 0:
        args.noise_type = 'clean'
        
    print(args)

    # Prepare Clean Data
    features, adj, labels, idx_train, idx_val, idx_test, nfeat, nclass = load_data(args, device)

    # Generate Perturbed Data
    if args.noise_type != 'clean': 
        adj, labels = load_perturbed_data(args, features, adj, labels, idx_train, idx_val, idx_test, device)
        
    classes, counts = np.unique(labels[idx_train].cpu().numpy(), return_counts=True)
    print(idx_train.shape[0], idx_val.shape[0], idx_test.shape[0], classes.shape[0], dict(zip(classes, counts)))

    # Model Initialization
    model = HRGNN(args, nfeat, nclass, device)
    model.fit(features, adj, labels, idx_train, idx_val, idx_test, args.epochs)
    acc_val = model.test(labels, idx_val)
    acc_test = model.test(labels, idx_test)
    
    print(f'Noise type: {args.noise_type}, Val accuracy: {acc_val:.4f}, Test accuracy: {acc_test:.4f}')
    
    return acc_val, acc_test

if __name__ == '__main__':
    main(args)