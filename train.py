import os
import argparse
import logging
import torch
from scipy.sparse import SparseEfficiencyWarning
import numpy as np
import random

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.data_utils import process_files
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl

from model.graph_classifier import GraphClassifier as dgl_model

from managers.evaluator import Evaluator
from managers.trainer import Trainer

from warnings import simplefilter

import wandb



def main(params):

    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    #need to add similar edge before extract subgraph


    params.db_path = os.path.join(params.main_dir, f'data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}')

    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    # train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', params.file_paths,
    #                         add_traspose_rels=params.add_traspose_rels,
    #                         num_neg_samples_per_link=params.num_neg_samples_per_link)
    # valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', params.file_paths,
    #                         add_traspose_rels=params.add_traspose_rels,
    #                         num_neg_samples_per_link=params.num_neg_samples_per_link)

    # Preprocess data once
    ssp_graph, __, __, __, id2entity, id2relation, bert_embeddings = process_files(params.file_paths, None)

    # Create train and valid datasets using the preprocessed data
    train = SubgraphDataset(params.db_path, 'train_pos', 'train_neg', ssp_graph, id2entity, id2relation,
                            bert_embeddings,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link)

    valid = SubgraphDataset(params.db_path, 'valid_pos', 'valid_neg', ssp_graph, id2entity, id2relation,
                            bert_embeddings,
                            add_traspose_rels=params.add_traspose_rels,
                            num_neg_samples_per_link=params.num_neg_samples_per_link)

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.inp_dim = train.n_feat_dim

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label

    graph_classifier = initialize_model(params, dgl_model, params.load_model)

    logging.info(f"Device: {params.device}")
    logging.info(f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    valid_evaluator = Evaluator(params, graph_classifier, valid)

    trainer = Trainer(params, graph_classifier, train, valid_evaluator)

    logging.info('Starting training with full batch...')

    trainer.train()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='TransE model')

    # Experiment setup params
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")
    parser.add_argument("--dataset", "-d", type=str,
                        help="Dataset string")
    parser.add_argument("--gpu", "-g", type=int, default=-1,
                        help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true',
                        help='Load existing model?')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")
    # Training regime params
    parser.add_argument("--num_epochs", "-ne", type=int, default=100,
                        help="Learning rate of the optimizer")
    parser.add_argument("--eval_every", type=int, default=3,
                        help="Interval of epochs to evaluate the model?")
    parser.add_argument("--eval_every_iter", type=int, default=150,
                        help="Interval of iterations to evaluate the model?")
    parser.add_argument("--save_every", type=int, default=3,
                        help="Interval of epochs to save a checkpoint of the model?")
    parser.add_argument("--early_stop", type=int, default=3,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate of the optimizer")
    parser.add_argument("--clip", type=int, default=1000,
                        help="Maximum gradient norm allowed")
    parser.add_argument("--l2", type=float, default= 1e-2,
                        help="Regularization constant for GNN weights")
    parser.add_argument("--margin", type=float, default=10,
                        help="The margin between positive and negative samples in the max-margin loss")

    # Data processing pipeline params
    parser.add_argument("--max_links", type=int, default=1000000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2,
                        help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=None,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--model_type', '-m', type=str, choices=['ssp', 'dgl'], default='dgl',
                        help='what format to store subgraphs in for model')
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_neg_samples_per_link", '-neg', type=int, default=10,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of dataloading processes")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')

    # Model params
    parser.add_argument("--rel_emb_dim", "-r_dim", type=int, default=32,
                        help="Relation embedding size")
    parser.add_argument("--attn_rel_emb_dim", "-ar_dim", type=int, default=32,
                        help="Relation embedding size for attention")
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", "-l", type=int, default=3,
                        help="Number of GCN layers")
    parser.add_argument("--num_bases", "-b", type=int, default=4,
                        help="Number of basis functions to use for GCN weights")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout rate in GNN layers")
    parser.add_argument("--edge_dropout", type=float, default=0.5,
                        help="Dropout rate in edges of the subgraphs")
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', action='store_true',
                        help='whether to have attn in model or not')

    parser.add_argument('--six_mode', '-six', type=bool, default=False,
                        help='whether to start the six mode')
    parser.add_argument('--no_jk', action='store_true',
                        help='Disable JK connection')
    parser.add_argument("--loss", type=int, default=0,
                        help='0,1,2 correspond to oriloss, absloss, BCEloss ')
    parser.add_argument('--critic', type=int, default=0,
                        help='0,1,2 correspond to auc, auc_pr, mrr')
    parser.add_argument('--epoch', type=int, default=0,
                        help='to record epoch')
    # parser.add_argument('--no_train', default=False, action="store_true")
    # parser.add_argument('--ablation', type=int, default=0,
    #                     help='0,1,2,3 correspond to normal, no-sub, no-ent, only-rel')

    parser.add_argument('--seed', default=41504, type=int, help='Seed for randomization')

    

    params = parser.parse_args()
    initialize_experiment(params, __file__)

    params.file_paths = {
        'train': os.path.join('data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join('data/{}/{}.txt'.format(params.dataset, params.valid_file))
    }

    np.random.seed(params.seed)
    random.seed(params.seed)
    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
        torch.cuda.manual_seed_all(params.seed)
        torch.backends.cudnn.deterministic = True
    else:
        params.device = torch.device('cpu')

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    # wandb.login(key='3d375be661f64b0eb6a5e10c9d5b1748d080d4a5')
    # writer = wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="WN18RR_TACT",
    #
    #     # track hyperparameters and run metadata
    #     config={
    #         "learning_rate": 0.005,
    #         "architecture": "gragh_classifier",
    #         "dataset": "WN18RR_v1",
    #         "epochs": 10,
    #     }
    # )
    main(params)
