import os
import pdb
import random

import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
from utils.graph_utils import ssp_multigraph_to_dgl, incidence_matrix
import torch
import dgl
import dgl.function as fn
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
import networkx as nx
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel


class Inference:
    def __init__(self, model_path, max_length=64, bert_name='bert-base-uncased', cuda=True, dropout=0.4):
        self.model_path = model_path
        self.max_length = max_length
        self.bert_name = bert_name
        self.cuda = cuda
        self.dropout = dropout

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
        self.encoder = AutoModel.from_pretrained(self.bert_name)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.e1_start_id = self.tokenizer.convert_tokens_to_ids('<e1>')
        self.e1_end_id = self.tokenizer.convert_tokens_to_ids('</e1>')
        self.e2_start_id = self.tokenizer.convert_tokens_to_ids('<e2>')
        self.e2_end_id = self.tokenizer.convert_tokens_to_ids('</e2>')

        self.model = SeqClsModel(self.dropout, 2, self.encoder)
        self.model.load_state_dict(torch.load(self.model_path))
        self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def predict(self, h, r, t):
        sent = ['<e1>'] + h.strip().split() + ['</e1>'] + \
               r.strip().split() + \
               ['<e2>'] + t.strip().split() + ['</e2>']
        tokenized = self.tokenizer(
            sent,
            padding=True,
            truncation=True,
            return_tensors='pt',
            is_split_into_words=True,
            max_length=self.max_length,
        )
        input_ids = tokenized['input_ids'].squeeze().tolist()
        e1_start = torch.tensor(input_ids.index(self.e1_start_id)).unsqueeze(0)
        e1_end = torch.tensor(input_ids.index(self.e1_end_id)).unsqueeze(0)
        e2_start = torch.tensor(input_ids.index(self.e2_start_id)).unsqueeze(0)
        e2_end = torch.tensor(input_ids.index(self.e2_end_id)).unsqueeze(0)
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        with torch.no_grad():
            logit = self.model(tokenized, cues=(e1_start, e1_end, e2_start, e2_end))
            logit = torch.argmax(logit, dim=-1)
            logit = logit.detach().cpu().numpy().tolist()
        return logit[0]


class SeqClsModel(nn.Module):
    def __init__(self, dropout, num_classes, encoder):
        super(SeqClsModel, self).__init__()

        self.dropout = dropout
        self.num_classes = num_classes
        self.encoder = encoder

        self.hidden_size = self.encoder.config.hidden_size
        self.in_features = 2 * self.hidden_size
        self.fc = nn.Sequential(
            nn.LayerNorm(self.in_features),
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features=self.in_features, out_features=self.num_classes),
        )

    def forward(self, x, cues):
        out = self.encoder(**x).last_hidden_state
        e1_start, e1_end, e2_start, e2_end = cues
        embed_e1 = self._get_pooled_embedding(out, e1_start, e1_end)
        embed_e2 = self._get_pooled_embedding(out, e2_start, e2_end)
        out = torch.cat((embed_e1, embed_e2), dim=1)
        out = self.fc(out)
        return out

    @staticmethod
    def _get_pooled_embedding(embeddings, starts, ends):
        features = []
        for embed, start, end in zip(embeddings, starts, ends):
            pooled_embeddings = torch.max(embed[start:end + 1], dim=0, keepdim=True).values
            features.append(pooled_embeddings)
        return torch.cat(features)


def file_write(file_path, data):
    with open(file_path, 'w+', encoding='utf-8') as f:
        for d in data:
            f.writelines(f'[{d[0]} {d[1]} {d[2]}]\n')
    print(f'Write file {file_path} done!')


def plot_rel_dist(adj_list, filename):
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    fig.savefig(filename, dpi=fig.dpi)


def dense_datasets(file_path):
    entity2id = {}
    relation2id = {}
    entity_frequency = {}
    triplets = {}

    ent = 0
    rel = 0
    frequency = 1

    data = []
    with open(file_path, encoding='utf-8') as f:
        raw_data = f.readlines()
    file_data = []
    for instance in raw_data:
        instance = instance.strip()
        if instance:
            instance = instance.split('\t')
            if len(instance) == 3:
                file_data.append(instance)
    print(file_data)

    for triplet in file_data:
        if triplet[1] not in entity2id:
            entity2id[triplet[1]] = ent
            entity_frequency[entity2id[triplet[1]]] = frequency
            ent += 1
        else:
            entity_frequency[entity2id[triplet[1]]] += 1
        if triplet[2] not in entity2id:
            entity2id[triplet[2]] = ent
            entity_frequency[entity2id[triplet[2]]] = frequency
            ent += 1
        else:
            entity_frequency[entity2id[triplet[2]]] += 1
        if triplet[0] not in relation2id:
            relation2id[triplet[0]] = rel
            rel += 1

        # Save the triplets corresponding to only the known relations
        if triplet[0] in relation2id:
            data.append([entity2id[triplet[1]], entity2id[triplet[2]], relation2id[triplet[0]]])

    triplets['train'] = np.array(data, dtype=object)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}


    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                   shape=(len(entity2id), len(entity2id))))

    # ADD SIM EDGES
    average_degree = sum(entity_frequency.values()) // len(entity2id) + 1
    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(adj_list[0].shape[0])))
    for rel, adj in enumerate(adj_list):
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # origin_graph = ssp_multigraph_to_dgl(adj_list)
    origin_graph = dgl.from_networkx(g_nx, edge_attrs=['type'])
    init_ent = torch.zeros((origin_graph.num_nodes(), 8))
    gain = torch.nn.init.calculate_gain('relu')
    torch.nn.init.xavier_normal_(init_ent, gain=gain)
    origin_graph.ndata['init'] = init_ent

    def add(nodes):
        return {'i': nodes.data['init'] + nodes.data['agg']}

    origin_graph.update_all(fn.copy_u('init', 'm'), fn.sum('m', 'agg'), add)
    print(origin_graph.ndata['i'])
    print(origin_graph.ndata['i'].size())
    print("number of original triplets: ", len(triplets['train']))
    init_ent_emb = origin_graph.ndata['i'].numpy()
    sim_matrix = cosine_similarity(init_ent_emb, init_ent_emb)
    sim = len(id2relation)
    triplets_train = []
    tau = 0.97
    indices = np.where(sim_matrix > tau)
    potential_ind = []
    potential_ind.extend([
        [i, j, sim]
        for i, j in tqdm(zip(indices[0], indices[1]), total=len(indices[0]), desc='Process')
        if i < j and not origin_graph.has_edges_between(i, j) and entity_frequency[i] < average_degree and entity_frequency[j] < average_degree
    ])
    print("number of add edges triplets: ", len(potential_ind))
    sim_edge = "sim"

    with open(file_path, 'a', encoding='utf-8') as f:
        for tri in potential_ind:
            f.writelines(f'{sim_edge}\t{id2entity.get(tri[0])}\t{id2entity.get(tri[1])}\n')
    print(f'Write file {file_path} done!')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你使用多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_files(files, saved_relation2id=None):
    set_seed(42)  # 设置随机种子

    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = []
        with open(file_path, encoding='utf-8') as f:
            raw_data = f.readlines()
        file_data = []
        for instance in raw_data:
            instance = instance.strip()
            if instance:
                instance = instance.split('\t')
                if len(instance) == 3:
                    file_data.append(instance)
        # print(file_data)

        for triplet in file_data:
            if triplet[1] not in entity2id:
                entity2id[triplet[1]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[0] not in relation2id:
                relation2id[triplet[0]] = rel
                rel += 1

            # Save the triplets corresponding to only the known relations
            if triplet[0] in relation2id:
                data.append([entity2id[triplet[1]], entity2id[triplet[2]], relation2id[triplet[0]]])
        triplets[file_type] = np.array(data, dtype=object)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to each relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8),
                                    (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))),
                                   shape=(len(entity2id), len(entity2id))))

    # 加载预训练的BERT模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    new_embedding_dim = 122

    # 切换到评估模式
    model.eval()

    # 创建全连接层来调整嵌入维度，并固定其权重
    linear_layer = nn.Linear(model.config.hidden_size, new_embedding_dim)
    with torch.no_grad():
        torch.nn.init.xavier_uniform_(linear_layer.weight)
        torch.nn.init.constant_(linear_layer.bias, 0.0)

    def generate_embeddings(id2entity):
        def get_bert_embedding(text):
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # 获取[CLS] token的嵌入
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            # 使用固定的全连接层来调整嵌入维度
            transformed_embeddings = linear_layer(cls_embedding)
            return transformed_embeddings.squeeze()

        embedding_dict = {node_id: get_bert_embedding(entity) for node_id, entity in id2entity.items()}
        return embedding_dict

    embedding_dict = generate_embeddings(id2entity)

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, embedding_dict


def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')
