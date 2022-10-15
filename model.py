import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import trange

from utils import add_edges
from deeprobust.graph.utils import accuracy


class HRGNN(nn.Module):
    def __init__(self, args, nfeat, nclass, device=None) -> None:
        super().__init__()
        
        self.device = device
        self.nfeat = nfeat
        self.n_class = nclass
        self.nhid = args.hidden
        
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        
        self.k = args.k
        self.threshold = args.threshold
        self.num_fakes = args.fake_nodes
        self.iteration = args.iteration
        self.add_labels = args.add_labels
        
        self.init_layers()
        self.pseudo_nodes_list = []
        
        self.to(device)
        
    def init_layers(self):
        self.edge_predictor = MLP(self.nfeat, self.nhid, self.nhid)
        self.node_classifier = GCN(self.nfeat, self.nhid, self.nhid, self.n_class, mask=True)
        self.output = nn.Linear(self.nhid, self.n_class)

        self.decoder = GCN(self.nhid, self.nhid, self.nfeat, self.nhid, mask=True)
        self.encoder_to_decoder = nn.Linear(self.nhid, self.nhid)

    def init_label_dict(self, labels, idx_train):

        train_y = labels[idx_train].cpu().numpy()
        self.label_ratio = {}
        for label in set(train_y):
            self.label_ratio[label] = sum(train_y==label) / len(train_y)

        self.training_ratio = labels.shape[0] / idx_train.shape[0]

    def forward_classifier(self, model, x, labels=None, mask=None):
        if labels == None:
            model.eval()
            logit = model(*x)
            return logit[:self.n_real]
        else:
            model.train()
            logit = model(*x)
            return logit, F.nll_loss(logit[mask], labels[mask])

    def create_fake_nodes(self, num_fakes, x, labels, idx_train, mode='normalize'):
        if type(num_fakes) == int:
            num_fakes = [num_fakes] * self.n_class
        
        elif type(num_fakes) == list:
            assert self.n_class == len(num_fakes), "the length of 'num_fakes' must equal to n_class"

        x = x[idx_train]
        labels = labels[idx_train].cpu()
        x_fake = []
        labels_fake = []
        edge_fake = []

        i = self.n_real
        for c, n_fake in enumerate(num_fakes):
            x_class = x[labels==c]
            random_weight = torch.rand(n_fake, x_class.size(0)).to(self.device)
            if mode == 'normalize':
                w = F.normalize(random_weight, p=1)
            elif mode == 'softmax':
                w = F.softmax(random_weight, 1)
            x_fake.append(torch.mm(w, x_class))
            labels_fake.append(torch.LongTensor([c]*n_fake))

        x_fake = torch.cat(x_fake)
        label_fake = torch.cat(labels_fake).to(self.device)

        return x_fake, label_fake, torch.arange(x_fake.size(0)).to(self.device)+self.n_real

    def create_mask_nodes(self, n, mask_rate):
        perm = torch.randperm(n)
        if mask_rate > 1:
            num_mask_nodes = mask_rate
        else:
            num_mask_nodes = int(self.n_real * mask_rate)

        return perm[:num_mask_nodes], perm[num_mask_nodes:]
    
    def sim_loss(self, real_features, pred_features, loss_type='cos', alpha=1):
        if loss_type == 'mse':
            loss = F.mse_loss(real_features, pred_features)
        else:
            x = F.normalize(real_features, p=2, dim=-1)
            y = F.normalize(pred_features, p=2, dim=-1)
            loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
            loss = loss.mean()
        return loss
    
    def recons_loss(self, embeddings, edge_index, edge_weight=None):
        mask_nodes, unmask_nodes = self.create_mask_nodes(n=self.n_real, mask_rate=0.2)
        
        self.encoder_to_decoder.train()
        self.decoder.train()
        z = self.encoder_to_decoder(embeddings)
        reconst = self.decoder.get_embeds(z, edge_index, edge_weight, mask_nodes=mask_nodes, mask_embedding=True)
        loss = self.sim_loss(self.x[mask_nodes], reconst[mask_nodes], loss_type='cos')

        return loss
    
    def class_loss(self, embeddings, labels, idx_train):
        logit = F.log_softmax(self.output(embeddings), -1)
        loss = F.nll_loss(logit[idx_train], labels[idx_train])
        
        return loss

    def fit(self, x, adj, labels, idx_train, idx_val=None, idx_test=None, epochs=200):
        idx_train = torch.LongTensor(idx_train).to(self.device)
        self.init_label_dict(labels, idx_train)

        self.n_real = x.size(0)
        optimizer = torch.optim.Adam(list(self.node_classifier.parameters())+
                                     list(self.edge_predictor.parameters())+
                                     list(self.output.parameters())+
                                     list(self.decoder.parameters())+
                                     list(self.encoder_to_decoder.parameters()),
                                     lr=self.lr, weight_decay=self.weight_decay)

        real_edge_index = adj.nonzero().T
        real_edge_weight = adj[tuple(real_edge_index)]
        
        fake_x, fake_labels, fake_nodes = self.create_fake_nodes(self.num_fakes, x, labels, idx_train)

        train_nodes = idx_train.clone()
        train_labels = labels.clone()

        self.x = torch.cat([x, fake_x])
        train_labels= torch.cat([train_labels, fake_labels])
        train_nodes = torch.cat([train_nodes, fake_nodes])


        best_acc = 0
        for i in trange(self.iteration):
            for epoch in range(epochs):
                optimizer.zero_grad()
                self.edge_predictor.train()
                embeddings = self.edge_predictor.get_embeds(self.x)
                
                loss_fr = self.recons_loss(embeddings, real_edge_index) 
                loss_nc = self.class_loss(embeddings, train_labels, train_nodes)
                
                train_label_mask = torch.logical_and(torch.isin(real_edge_index[0], train_nodes), torch.isin(real_edge_index[1], train_nodes))
                label_mask_ = train_labels[real_edge_index[0]] != train_labels[real_edge_index[1]]
                label_mask = ~torch.logical_and(label_mask_, train_label_mask)
                new_edge_index, new_edge_weight = real_edge_index[:, label_mask], real_edge_weight[label_mask]

                knn_edge_index, knn_edge_weight = self.knn(embeddings, self.k, 1000, self.device)
                fake_edge_index, fake_edge_weight, edge_mask = add_edges(knn_edge_index, knn_edge_weight, train_labels, train_nodes, self.n_real, mode='mix', threshold=self.threshold)
                
                self.edge_index = torch.cat([new_edge_index, fake_edge_index], -1)
                self.edge_weight = torch.cat([new_edge_weight, fake_edge_weight])

                y_pred, loss_g = self.forward_classifier(self.node_classifier, (self.x, self.edge_index, self.edge_weight), 
                                                         train_labels, train_nodes)
                loss = loss_g + loss_nc + loss_fr
                loss.backward()
                optimizer.step()
                
                
                # evaluate
                y_pred = self.forward_classifier(self.node_classifier, (self.x, self.edge_index, self.edge_weight))
                accs = []
                for mask in [train_nodes[train_nodes<self.n_real], idx_val, idx_test]:
                    pred = y_pred[mask].max(1)[1]
                    acc = pred.eq(labels[mask]).sum().item() / len(mask)
                    accs.append(acc)

                if accs[1] > best_acc:
                    best_acc = accs[1]
                    best_test_acc = accs[2]
                    best_node_classifier_wts = copy.deepcopy(self.node_classifier.state_dict())
                    best_edge_predictor_wts = copy.deepcopy(self.edge_predictor.state_dict())
                    best_edge_index = self.edge_index.clone()
                    best_edge_weight = self.edge_weight.clone()

            pseudo_nodes, pseudo_labels = self.add_nodes(train_nodes, y_pred)
            train_labels[pseudo_nodes] = pseudo_labels
            train_nodes = torch.cat([train_nodes, pseudo_nodes])
            self.pseudo_nodes_list.extend(pseudo_nodes.tolist())

        self.restore_all(best_edge_predictor_wts, best_node_classifier_wts, best_edge_index, best_edge_weight)

    def restore_all(self, edge_predictor_wts, node_classifier_wts, edge_index, edge_weight):

        self.edge_predictor.load_state_dict(edge_predictor_wts)
        self.node_classifier.load_state_dict(node_classifier_wts)

        self.edge_predictor.eval()
        self.node_classifier.eval()

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        
    def add_nodes(self, train_nodes, y_pred):
        n = self.add_labels
        mask = torch.isin(torch.arange(self.n_real).to(self.device), train_nodes)
        unlabel_nodes = torch.where(~mask)[0]

        new_nodes = []
        new_labels = []

        unlabel_logit, unlabel_pseudo = y_pred[unlabel_nodes].max(-1)

        for c, r in self.label_ratio.items():
            n_class = int(r*n)
            idx_class = torch.where(unlabel_pseudo == c)[0]

            if len(idx_class) < n_class:
                idx = idx_class[unlabel_logit[idx_class].topk(len(idx_class))[1]]
            else:
                idx = idx_class[unlabel_logit[idx_class].topk(n_class)[1]]

            new_nodes.append(unlabel_nodes[idx])
            new_labels.append(unlabel_pseudo[idx])

        new_nodes = torch.cat(new_nodes)
        new_labels = torch.cat(new_labels)

        return new_nodes, new_labels
        
    def forward(self, x, adj):
        if x == None and adj == None:
            x = self.x
            edge_index = self.edge_index
            edge_weight = self.edge_weight
        else:
            edge_index = adj.nonzero().T
            edge_weight = adj[tuple(edge_index)]
            
        y_pred = self.forward_classifier(self.node_classifier, (x, edge_index, edge_weight))

        return y_pred

    def predict(self, x=None, adj=None):
        return self.forward(x, adj)
    
    def test(self, labels, idx_test):
        self.eval()
        output = self.predict().cpu()
        acc = accuracy(output[idx_test], labels[idx_test]).cpu().item()
        
        return acc

    def get_embed(self, x, adj):
        self.node_classifier.eval()
        s_embeds = self.node_classifier.get_embeds(self.x, self.edge_index, self.edge_weight)
        
        return s_embeds

    def knn(self, Z, k, b, device, normalize=True):
        X = Z[:self.n_real] # Real Nodes
        Y = Z[self.n_real:] # Fake Nodes 
        if normalize:
            X = F.normalize(X, dim=1, p=2)
            Y = F.normalize(Y, dim=1, p=2)

        index = 0
        values = torch.zeros(X.shape[0] * (k + 1), device=device)
        rows = torch.zeros(X.shape[0] * (k + 1), device=device)
        cols = torch.zeros(X.shape[0] * (k + 1), device=device)

        while index < X.shape[0]:
            if (index + b) > (X.shape[0]):
                end = X.shape[0]
            else:
                end = index + b
            sub_tensor = X[index:index + b]
            similarities = torch.mm(sub_tensor, Y.t())
            vals, inds = similarities.topk(k=k + 1, dim=-1)
            values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
            cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
            rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
            index += b

        rows = rows.long()
        cols = cols.long() + self.n_real

        rows_ = torch.cat((rows, cols))
        cols_ = torch.cat((cols, rows))
        edge_index = torch.stack([rows_, cols_])
        edge_weight = torch.cat((values, values)).relu()

        return edge_index, edge_weight

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Linear(in_dim, hid_dim)
        self.conv2 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        x = self.get_embeds(x)

        return F.log_softmax(x, dim=1)

    def get_embeds(self, x):
        x = F.relu(self.conv1(x))

        return self.conv2(x)

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, hid_dim1, out_dim, mask=False):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim1)
        self.output = nn.Linear(hid_dim1, out_dim)
        if mask:
            self.mask_embedding = torch.nn.Parameter(torch.zeros(1,in_dim))

    def forward(self, x, edge_index, edge_weight=None):
        x = self.get_embeds(x, edge_index, edge_weight)
        x = self.output(x)

        return F.log_softmax(x, dim=1)

    def process_mask(self, x, mask_nodes, mask_embedding):
        if mask_nodes != None:
            if mask_embedding:
                out_x = x.clone()
            else:
                out_x = x
            out_x[mask_nodes] = 0
            if mask_embedding:
                out_x[mask_nodes] += self.mask_embedding
        else:
            out_x = x

        return out_x

    def get_embeds(self, x, edge_index, edge_weight=None, mask_nodes=None, mask_embedding=False):
        x = self.process_mask(x, mask_nodes, mask_embedding)
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)

        return x

    