import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn import metrics


class Trainer():
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        print('Total number of parameters: %d' % sum(map(lambda x: x.numel(), model_params)))

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(model_params, lr=params.lr, momentum=params.momentum, weight_decay=self.params.l2)
        elif params.optimizer == "Adam":
            self.optimizer = optim.Adam(model_params, lr=params.lr, weight_decay=self.params.l2)
        elif params.optimizer == "AdamW":
            self.optimizer = optim.AdamW(model_params, lr=params.lr, weight_decay=self.params.l2)

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction='sum')
        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(self.train_data, batch_size=self.params.batch_size, shuffle=True, num_workers=0,
                                collate_fn=self.params.collate_fn)
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())
        for b_idx, batch in enumerate(dataloader):
            data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
            self.optimizer.zero_grad()

            score_pos = self.graph_classifier(data_pos)
            score_neg = self.graph_classifier(data_neg)

            loss = self.criterion(score_pos.squeeze(), score_neg.view(len(score_pos), -1).mean(dim=1),
                                  torch.Tensor([1] * score_pos.shape[0]).to(device=self.params.device))
            loss.backward(retain_graph=True)
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                all_scores += score_pos.squeeze().detach().cpu().tolist() + score_neg.squeeze().detach().cpu().tolist()
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                loss_item = loss.detach().cpu().item()
                total_loss += loss_item

        auc_roc = round(metrics.roc_auc_score(all_labels, all_scores), 4)
        auc_pr = round(metrics.average_precision_score(all_labels, all_scores), 4)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))

        # 在每个epoch结束后进行验证评估
        if self.valid_evaluator:
            tic = time.time()
            result = self.valid_evaluator.eval()

            if result['auc_roc'] > self.best_metric:
                self.save_classifier()
                self.best_metric = result['auc_roc']
                self.not_improved_count = 0
            else:
                self.not_improved_count += 1
                print(f"Validation not improved count: {self.not_improved_count}")
                if self.not_improved_count >= self.params.early_stop:
                    print(f"Validation performance didn't improve for {self.params.early_stop} epochs. Training stops.")
                    return total_loss, auc_roc, auc_pr, weight_norm, True  # 训练应停止

        return total_loss, auc_roc, auc_pr, weight_norm, False  # 继续训练

    def train(self):
        self.reset_training_state()
        for epoch in range(1, self.params.num_epochs + 1):
            loss, auc_roc, auc_pr, weight_norm, should_stop = self.train_epoch()
            logging.info(
                f'Epoch {epoch} with loss: {loss}, training auc_roc: {auc_roc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}')
            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))
            torch.cuda.empty_cache()
            if should_stop:
                print(f"Training stopped early at epoch {epoch}")
                break

    def save_classifier(self):
        torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'best_graph_classifier.pth'))
        logging.info(f'Better models found w.r.t validation performance. Saved it!')
