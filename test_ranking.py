import os
import random
import argparse
import logging
import json
import time

import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx
import torch
import torch.nn as nn
import numpy as np
import dgl
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel
from concurrent.futures import ThreadPoolExecutor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你使用多个GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_files(files, saved_relation2id, add_traspose_rels):
    '''
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    '''
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

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['graph'][:, 2] == i)
        adj_list.append(ssp.csc_matrix((np.ones(len(idx), dtype=np.uint8), (
            triplets['graph'][:, 0][idx].squeeze(1), triplets['graph'][:, 1][idx].squeeze(1))),
                                       shape=(len(entity2id), len(entity2id))))

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

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

    # 批处理大小
    batch_size = 32

    def get_bert_embeddings_batch(texts):
        inputs = tokenizer(texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        # 获取[CLS] token的嵌入
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        # 使用固定的全连接层来调整嵌入维度
        transformed_embeddings = linear_layer(cls_embeddings)
        return transformed_embeddings.detach()

    def generate_embeddings(id2entity):
        node_ids = list(id2entity.keys())
        entities = list(id2entity.values())

        embedding_dict = {}
        total_batches = (len(entities) + batch_size - 1) // batch_size  # 计算总批次数

        # 使用线程池并行处理
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(entities), batch_size):
                batch_entities = entities[i:i + batch_size]
                futures.append(executor.submit(get_bert_embeddings_batch, batch_entities))

            for i, future in enumerate(tqdm(futures, desc="Generating embeddings", total=total_batches)):
                batch_embeddings = future.result()
                batch_node_ids = node_ids[i * batch_size:(i + 1) * batch_size]
                for j, node_id in enumerate(batch_node_ids):
                    embedding_dict[node_id] = batch_embeddings[j]
        return embedding_dict

    embedding_dict = generate_embeddings(id2entity)
    print("nodes: ", id2entity.get(0))
    print("bert_embedding: ", embedding_dict.get(0))

    return adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation, embedding_dict


def intialize_worker(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, bert_embedding):
    global model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, bert_embedding_
    model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_, bert_embedding_= model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, bert_embedding


def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    for i, (head, tail, rel) in enumerate(zip(heads, tails, rels)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        while len(neg_triplet['head'][0]) < num_samples:
            neg_head = head
            neg_tail = np.random.choice(n)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        while len(neg_triplet['tail'][0]) < num_samples:
            neg_head = np.random.choice(n)
            neg_tail = tail
            # neg_head, neg_tail, rel = np.random.choice(n), np.random.choice(n), np.random.choice(r)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):

    n, r = adj_list[0].shape[0], len(adj_list)
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]

    neg_triplets = []
    print('sampling negative triplets...')
    for i, (head, tail, rel) in tqdm(enumerate(zip(heads, tails, rels)), total=len(heads)):
        neg_triplet = {'head': [[], 0], 'tail': [[], 0]}
        neg_triplet['head'][0].append([head, tail, rel])
        for neg_tail in range(n):
            neg_head = head

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['head'][0].append([neg_head, neg_tail, rel])

        neg_triplet['tail'][0].append([head, tail, rel])
        for neg_head in range(n):
            neg_tail = tail

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_triplet['tail'][0].append([neg_head, neg_tail, rel])

        neg_triplet['head'][0] = np.array(neg_triplet['head'][0])
        neg_triplet['tail'][0] = np.array(neg_triplet['tail'][0])

        neg_triplets.append(neg_triplet)

    return neg_triplets


def incidence_matrix(adj_list):
    '''
    adj_list: List of sparse adjacency matrices
    '''

    rows, cols, dats = [], [], []
    dim = adj_list[0].shape
    for adj in adj_list:
        adjcoo = adj.tocoo()
        rows += adjcoo.row.tolist()
        cols += adjcoo.col.tolist()
        dats += adjcoo.data.tolist()
    row = np.array(rows)
    col = np.array(cols)
    data = np.array(dats)
    return ssp.csc_matrix((data, (row, col)), shape=dim)


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    Modified from dgl.contrib.data.knowledge_graph to node accomodate sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None, node_information=None, max_node_label_value=None):
    # extract the h-hop enclosing subgraphs around link 'ind'
    A_incidence = incidence_matrix(A_list)
    A_incidence += A_incidence.T

    # could pack these two into a function
    root1_nei = get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
    root2_nei = get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)

    subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
    subgraph_nei_nodes_un = root1_nei.union(root2_nei)

    # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
    if enclosing_sub_graph:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
    else:
        subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)

    subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

    labels, enclosing_subgraph_nodes = node_label_new(incidence_matrix(subgraph), max_distance=h)

    pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
    pruned_labels = labels[enclosing_subgraph_nodes]

    if max_node_label_value is not None:
        pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

    return pruned_subgraph_nodes, pruned_labels


def remove_nodes(A_incidence, nodes):
    idxs_wo_nodes = list(set(range(A_incidence.shape[1])) - set(nodes))
    return A_incidence[idxs_wo_nodes, :][:, idxs_wo_nodes]


def node_label_new(subgraph, max_distance=1):
    roots = [0, 1]
    sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
    dist_to_roots = [np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7) for r, sg in enumerate(sgs_single_root)]
    dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

    # dist_to_roots[np.abs(dist_to_roots) > 1e6] = 0
    # dist_to_roots = dist_to_roots + 1
    target_node_labels = np.array([[0, 1], [1, 0]])
    labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

    enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
    # print(len(enclosing_subgraph_nodes))
    return labels, enclosing_subgraph_nodes


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {'type': rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    #g_dgl = dgl.DGLGraph(multigraph=True)
    g_dgl = dgl.from_networkx(g_nx, edge_attrs=['type'])
    # add node features
    if n_feats is not None:
        g_dgl.ndata['feat'] = torch.tensor(n_feats)

    return g_dgl


def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    # One hot encode the node label feature and concat to n_featsure
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1

    n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
    n_feats_tensor = torch.FloatTensor(n_feats)
    if bert_embedding_ is not None:
        embedding = []
        for node in subgraph.ndata[dgl.NID].tolist():
            embedding.append(bert_embedding_.get(node))
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

    return subgraph


def get_subgraphs(all_links, adj_list, dgl_adj_list, max_node_label_value):
    # dgl_adj_list = ssp_multigraph_to_dgl(adj_list)

    subgraphs = []
    r_labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        nodes, node_labels = subgraph_extraction_labeling((head, tail), rel, adj_list, h=params_.hop, enclosing_sub_graph=params_.enclosing_sub_graph, max_node_label_value=max_node_label_value)

        subgraph = dgl.node_subgraph(dgl_adj_list, nodes)
        #subgraph.edata['type'] = dgl_adj_list.edata['type'][dgl_adj_list.subgraph(nodes).parent_eid]
        subgraph.edata['label'] = torch.tensor(rel * np.ones(subgraph.edata['type'].shape), dtype=torch.long)

        edges_btw_roots = subgraph.edge_ids(0, 1, return_uv=True)[2]
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rel)
        if rel_link.squeeze().nelement() == 0:
            subgraph.add_edges(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(rel).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(rel).type(torch.LongTensor)

        # kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes] if kge_entity2id else None
        # n_feats = node_features[kge_nodes] if node_features is not None else None
        n_feats = None
        subgraph = prepare_features(subgraph, node_labels, max_node_label_value, n_feats)

        subgraphs.append(subgraph)
        r_labels.append(rel)

    batched_graph = dgl.batch(subgraphs)
    r_labels = torch.LongTensor(r_labels)

    return (batched_graph, r_labels)


def get_rank(neg_links):
    head_neg_links = neg_links['head'][0]
    head_target_id = neg_links['head'][1]

    if head_target_id != 10000:
        data = get_subgraphs(head_neg_links, adj_list_, dgl_adj_list_, params_.max_label_value)
        head_scores = model_(data).squeeze(1).detach().cpu().numpy()
        head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target_id) + 1
    else:
        head_scores = np.array([])
        head_rank = 10000

    tail_neg_links = neg_links['tail'][0]
    tail_target_id = neg_links['tail'][1]

    if tail_target_id != 10000:
        data = get_subgraphs(tail_neg_links, adj_list_, dgl_adj_list_, params_.max_label_value)
        tail_scores = model_(data).squeeze(1).detach().cpu().numpy()
        tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id) + 1
    else:
        tail_scores = np.array([])
        tail_rank = 10000

    return head_scores, head_rank, tail_scores, tail_rank


def save_negative_triples_to_file(neg_triplets, id2entity, id2relation):

    with open(os.path.join('../data', params.dataset, 'ranking_head.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['head'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

    with open(os.path.join('../data', params.dataset, 'ranking_tail.txt'), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet['tail'][0]:
                f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')


def main(params):
    model = torch.load(params.model_path, map_location='cpu')
    model.params.gpu = -1
    model.params.num_neg_samples_per_link = model.params.num_rels - 1
    params.max_label_value = np.array([2, 2])

    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation, bert_embedding = process_files(params.file_paths, model.relation2id, params.add_traspose_rels)


    node_features, kge_entity2id = (None, None)

    if params.mode == 'sample':
        neg_triplets = get_neg_samples_replacing_head_tail(triplets['links'], adj_list)
    elif params.mode == 'all':
        neg_triplets = get_neg_samples_replacing_head_tail_all(triplets['links'], adj_list)

    ranks = []
    all_head_scores = []
    all_tail_scores = []

    with mp.Pool(processes=None, initializer=intialize_worker, initargs=(model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id, bert_embedding)) as p:
        for head_scores, head_rank, tail_scores, tail_rank in tqdm(p.imap(get_rank, neg_triplets), total=len(neg_triplets)):
            ranks.append(head_rank)
            ranks.append(tail_rank)
            all_head_scores += head_scores.tolist()
            all_tail_scores += tail_scores.tolist()

    isHit1List = [x for x in ranks if x <= 1]
    isHit3List = [x for x in ranks if x <= 3]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) * 1.0 / len(ranks)
    hits_3 = len(isHit3List) * 1.0 / len(ranks)
    hits_10 = len(isHit10List) * 1.0 / len(ranks)

    mrr = np.mean(1.0 / np.array(ranks)).item()

    logger.info('test metrics. MRR: %.4f, H@1: %.4f, H@3: %.4f, H@10: %.4f' % (mrr, hits_1, hits_3, hits_10))

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Testing script for hits@10')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="fb_v2_margin_loss",
                        help="Experiment name. Log file with this name will be created")
    parser.add_argument("--dataset", "-d", type=str, default="FB237_v2",
                        help="Path to dataset")
    parser.add_argument("--mode", "-m", type=str, default="sample", choices=["sample", "all", "ruleN"],
                        help="Negative sampling mode")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument("--hop", type=int, default=2,
                        help="How many hops to go while eextracting subgraphs?")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='Whether to append adj matrix list with symmetric relations?')
    parser.add_argument('--final_model', action='store_true',
                        help='Disable CUDA')

    params = parser.parse_args()

    params.file_paths = {
        'graph': os.path.join('data', params.dataset, 'train.txt'),
        'links': os.path.join('data', params.dataset, 'test.txt')
    }

    # params.final_model = True
    if params.final_model:
        params.model_path = os.path.join('experiments', params.experiment_name, 'graph_classifier_chk.pth')
    else:
        params.model_path = os.path.join('experiments', params.experiment_name, 'best_graph_classifier.pth')

    file_handler = logging.FileHandler(os.path.join('experiments', params.experiment_name, f'log_rank_test_{time.time()}.txt'))
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v
                          in sorted(dict(vars(params)).items())))
    logger.info('============================================')

    main(params)
