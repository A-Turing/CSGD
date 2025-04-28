from .rgcn_model import RGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import torch.nn.functional as F
"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id

        self.relation_list = list(self.relation2id.values())
        self.no_jk = self.params.no_jk
        self.is_big_dataset = False

        self.gnn = RGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)
        self.rel_emb = nn.Embedding(self.params.num_rels, self.params.rel_emb_dim, sparse=False)

        torch.nn.init.normal_(self.rel_emb.weight)

        num_final_gcn_layer = self.params.num_gcn_layers
        if self.no_jk:
            num_final_gcn_layer = 1

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(3 * num_final_gcn_layer * self.params.emb_dim + self.params.rel_emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(num_final_gcn_layer * self.params.emb_dim + self.params.rel_emb_dim, 1)

    def forward(self, data):
        if self.params.gpu >= 0:
            device = torch.device('cuda:%d' % self.params.gpu)
        else:
            device = torch.device('cpu')

        g, rel_labels = data

        local_g = g.local_var()
        in_deg = local_g.in_degrees(range(local_g.number_of_nodes())).float()
        in_deg[in_deg == 0] = 1
        node_norm = 1.0 / in_deg
        local_g.ndata['norm'] = node_norm
        local_g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
        norm = local_g.edata['norm']

        if self.params.gpu >= 0:
            norm = norm.cuda(device=self.params.gpu)

        g.ndata['h'] = self.gnn(g, norm)

        g_out = mean_nodes(g, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        if self.no_jk:
            g_rep = torch.cat([g_out.view(-1, self.params.emb_dim),
                               head_embs.view(-1, self.params.emb_dim),
                               tail_embs.view(-1, self.params.emb_dim),
                               self.rel_emb(rel_labels)], dim=1)

        output = self.fc_layer(g_rep)
        return output


    @staticmethod
    def sparse_dense_mul(s, d):
        i = s._indices()
        v = s._values()
        dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
        return torch.sparse.FloatTensor(i, v * dv, s.size())

    @staticmethod
    def sparse_index_select(s, idx):
        indices_s = s._indices()
        indice_new_1 = torch.tensor([])
        indice_new_2 = torch.tensor([])
        num_i = 0.0
        for itm in idx:
            mask = (indices_s[0] == itm)
            indice_tmp_1 = torch.ones(sum(mask)) * num_i
            indice_tmp_2 = indices_s[1][mask].float()
            indice_new_1 = torch.cat((indice_new_1, indice_tmp_1), dim=0)
            indice_new_2 = torch.cat((indice_new_2, indice_tmp_2), dim=0)
            num_i = num_i + 1.0
        indices_new = torch.cat((indice_new_1.unsqueeze(0), indice_new_2.unsqueeze(0)), dim=0).long()

        return torch.sparse.FloatTensor(indices_new, torch.ones(indices_new.shape[1]),
                                        torch.Size((len(idx), s.shape[1])))