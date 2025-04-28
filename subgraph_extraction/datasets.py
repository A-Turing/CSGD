from torch.utils.data import Dataset
import timeit
import os
import logging
import lmdb
import numpy as np
import json
import pickle
import dgl
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
from utils.data_utils import process_files, save_to_file, plot_rel_dist, dense_datasets
from .graph_sampler import *
import pdb
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from tqdm import tqdm


def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None, max_label_value=None):
    testing = 'test' in splits
    dense_datasets(params.file_paths[splits[0]])
    print("ADDING SIM EDGES DONE!!")

    adj_list, triplets, entity2id, relation2id, id2entity, id2relation, bert_embeddings = process_files(
        params.file_paths,
        saved_relation2id)

    # plot_rel_dist(adj_list, os.path.join(params.main_dir, f'data/{params.dataset}/rel_dist.png'))
    data_path = os.path.join(f'data/{params.dataset}/relation2id.json')
    if not os.path.isdir(data_path) and not testing:
        with open(data_path, 'w') as f:
            json.dump(relation2id, f)

    graphs = {}

    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}

    # Sample train and valid/test links
    for split_name, split in graphs.items():
        print(f"Sampling negative links for {split_name}")
        split['pos'], split['neg'] = sample_neg(adj_list, split['triplets'], params.num_neg_samples_per_link,
                                                max_size=split['max_size'],
                                                constrained_neg_prob=params.constrained_neg_prob)

    if testing:
        directory = os.path.join(params.main_dir, '../data/{}/'.format(params.dataset))
        save_to_file(directory, f'negative_test_triples_by_{params.model}.txt', graphs['test']['neg'], id2entity,
                     id2relation)

    links2subgraphs(adj_list, graphs, params, max_label_value)


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    # def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None,
    #              add_traspose_rels=False, num_neg_samples_per_link=1):
    #
    #     self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
    #     self.db_pos = self.main_env.open_db(db_name_pos.encode())
    #     self.db_neg = self.main_env.open_db(db_name_neg.encode())
    #     self.num_neg_samples_per_link = num_neg_samples_per_link
    #
    #     ssp_graph, __, __, __, id2entity, id2relation, bert_embeddings = process_files(raw_data_paths,
    #                                                                                    included_relations)
    #
    #     print("ssp_graph", ssp_graph)
    #     print("id2entity", id2entity.get(0), id2entity.get(1000))
    #     print("bert_embeddings", bert_embeddings.get(0), bert_embeddings.get(1000))
    #     self.num_rels = len(ssp_graph)
    #     self.embeddings = bert_embeddings
    #
    #     # Add transpose matrices to handle both directions of relations.
    #     if add_traspose_rels:
    #         ssp_graph_t = [adj.T for adj in ssp_graph]
    #         ssp_graph += ssp_graph_t
    #
    #     # the effective number of relations after adding symmetric adjacency matrices and/or self connections
    #     self.aug_num_rels = len(ssp_graph)
    #     self.graph = ssp_multigraph_to_dgl(ssp_graph)
    #     self.ssp_graph = ssp_graph
    #
    #     self.max_n_label = np.array([0, 0])
    #     with self.main_env.begin() as txn:
    #         self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
    #         self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')
    #
    #         self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
    #         self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
    #         self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
    #         self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))
    #
    #         self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
    #         self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
    #         self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
    #         self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))
    #
    #         self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
    #         self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
    #         self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
    #         self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))
    #
    #     print(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")
    #
    #     with self.main_env.begin(db=self.db_pos) as txn:
    #         self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
    #     with self.main_env.begin(db=self.db_neg) as txn:
    #         self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
    #
    #     print("initialize subgraph:")
    #     for index in tqdm(range(self.num_graphs_pos + self.num_graphs_neg), desc="Processing"):
    #         self.__getitem__(0)

    def __init__(self, db_path, db_name_pos, db_name_neg, ssp_graph, id2entity, id2relation, bert_embeddings,
                 included_relations=None,
                 add_traspose_rels=False, num_neg_samples_per_link=1):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.num_neg_samples_per_link = num_neg_samples_per_link

        print("ssp_graph", ssp_graph)
        print("id2entity", id2entity.get(0), id2entity.get(1000))
        print("bert_embeddings", bert_embeddings.get(0), bert_embeddings.get(1000))
        self.num_rels = len(ssp_graph)
        self.embeddings = bert_embeddings

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)
        self.graph = ssp_multigraph_to_dgl(ssp_graph)
        self.ssp_graph = ssp_graph

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))[0]
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))[0]
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))[0]
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))[0]

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))[0]
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))[0]
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))[0]
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))[0]

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))[0]
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))[0]
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))[0]
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))[0]

        print(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        print("initialize subgraph:")
        for index in tqdm(range(self.num_graphs_pos + self.num_graphs_neg), desc="Processing"):
            self.__getitem__(0)

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos, self.embeddings)
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                subgraphs_neg.append(self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg, self.embeddings))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels, embeddings):
        subgraph = dgl.node_subgraph(self.graph, nodes)
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)
        edges_btw_roots = subgraph.edge_ids(0, 1, return_uv=True)[2]
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edges(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        n_feats = None
        subgraph = self._prepare_features_new(subgraph, n_labels, n_feats, embeddings)

        return subgraph

    def _prepare_features_new(self, subgraph, n_labels, n_feats=None, embeddings=None):
        # One hot encode the node label feature and concat to n_feature
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        n_feats_tensor = torch.FloatTensor(n_feats)
        if embeddings is not None:
            embedding = []
            for node in subgraph.ndata[dgl.NID].tolist():
                embedding.append(embeddings.get(node))
            embeddings_tensor = torch.stack(embedding)
            combined_embedding = torch.cat((n_feats_tensor, embeddings_tensor), dim=1)
        else:
            combined_embedding = n_feats_tensor
        subgraph.ndata['feat'] = combined_embedding

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)

        self.n_feat_dim = subgraph.ndata['feat'].shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph
