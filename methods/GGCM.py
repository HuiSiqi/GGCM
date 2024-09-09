import torch
import os
import torch.nn as nn
import numpy as np
from methods.meta_template_metaChannelAttention import MetaTemplate
from methods.gnn import GNN_nl
from methods import pikey_backbone as backbone
from torch.nn import functional as F
from utils import yellow_text, green_text, ensure_path, create_txt
from methods import gnn
import importlib
import random

class_categories = {}
class_categories['source'] = 64
class_categories['cub'] = 100
class_categories['cars'] = 97
class_categories['places'] = 182
class_categories['plantae'] = 99
class_categories['omniglot'] = 30
EPS = 0.00001

class SKL(torch.nn.Module):
    def __init__(self, temperature=5.0):
        super(SKL, self).__init__()
        self.temp = temperature
        self.KL = torch.nn.KLDivLoss()

    def forward(self, score_t, score_s):
        log_dist_t = F.log_softmax(score_t / self.temp, dim=1)
        dist_t = F.softmax(score_t / self.temp, dim=1)
        log_dist_s = F.log_softmax(score_s / self.temp, dim=1)
        dist_s = F.softmax(score_s / self.temp, dim=1)
        return 0.5 * (self.KL(log_dist_t, dist_s) + self.KL(log_dist_s, dist_t))

class GnnNetStudent(MetaTemplate):
    maml = True

    def __init__(self, model_func, args, n_way, n_support, tf_path=None, target_set='None'):
        super(GnnNetStudent, self).__init__(model_func, args, n_way, n_support, tf_path=tf_path)
        if self.maml:
            gnn.Gconv.maml = True
            gnn.Wcompute.maml = True
        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # metric function
        self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128),
                                nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml else nn.Sequential(
            backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
        self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)
        self.method = 'GnnNet'

        # define global fc classifiers
        self.classifier_source = nn.Linear(self.feat_dim,
                                           class_categories['source']) if not self.maml else backbone.Linear_fw(
            self.feat_dim, class_categories['source'])
        self.classifier_target = nn.Linear(self.feat_dim,
                                           class_categories[target_set]) if not self.maml else backbone.Linear_fw(
            self.feat_dim, class_categories[target_set])

        # define learnablMaskLayers
        self.MSA1 = torch.ones(1, 64, 1, 1)
        self.MSA2 = torch.ones(1, 128, 1, 1)
        self.MSA3 = torch.ones(1, 256, 1, 1)
        self.MSA4 = torch.ones(1, self.feat_dim, 1, 1)

        # fix label for training the metric function   1*nw(1 + ns)*nw
        support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
        support_label = torch.zeros(self.n_way * self.n_support, self.n_way).scatter(1, support_label, 1).view(
            self.n_way, self.n_support, self.n_way)

        support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)

        self.support_label = support_label.view(1, -1, self.n_way)
        self.skl = SKL()

    def cuda(self):
        self.feature.cuda()
        self.fc.cuda()
        self.gnn.cuda()
        self.support_label = self.support_label.cuda()
        self.classifier_source.cuda()
        self.classifier_target.cuda()
        self.MSA1 = self.MSA1.cuda()
        self.MSA2 = self.MSA2.cuda()
        self.MSA3 = self.MSA3.cuda()
        self.MSA4 = self.MSA4.cuda()
        return self

    def split_model_parameters(self, return_name=False):
        model_params = []
        model_params_name = []
        for n, p in self.named_parameters():
            n = n.split('.')
            model_params.append(p)
            model_params_name.append(n)
        if not return_name:
            return model_params
        else:
            return model_params, model_params_name

    def standard_path(self, x, **kwargs):
        # standard path

        self.x_fea_block1 = self.feature.forward_block1(x)

        self.x_fea_block2 = self.feature.forward_block2(self.x_fea_block1)

        self.x_fea_block3 = self.feature.forward_block3(self.x_fea_block2)

        self.x_fea_block4 = self.feature.forward_block4(self.x_fea_block3)

        self.x_fea = self.feature.forward_rest(self.x_fea_block4)

        self.x_fea_block5 = self.x_fea

        return self.x_fea

    def masked_path(self, x, reverse=False, detach=False):
        self.x_fea_block1 = self.feature.forward_block1(x)

        if 1 in self.args.meta_layers:
            x_fea_block1 = self.MSA1 * self.x_fea_block1
            self.x_fea_block2 = self.feature.forward_block2(x_fea_block1)
        else:
            self.x_fea_block2 = self.feature.forward_block2(self.x_fea_block1)

        if 2 in self.args.meta_layers:
            x_fea_block2 = self.MSA2 * self.x_fea_block2
            self.x_fea_block3 = self.feature.forward_block3(x_fea_block2)
        else:
            self.x_fea_block3 = self.feature.forward_block3(self.x_fea_block2)

        if 3 in self.args.meta_layers:
            x_fea_block3 = self.MSA3 * self.x_fea_block3
            self.x_fea_block4 = self.feature.forward_block4(x_fea_block3)
        else:
            self.x_fea_block4 = self.feature.forward_block4(self.x_fea_block3)

        if 4 in self.args.meta_layers:
            self.x_fea_block5 = self.MSA4 * self.x_fea_block4
            self.x_fea = self.feature.forward_rest(self.x_fea_block5)
        else:
            self.x_fea = self.feature.forward_rest(self.x_fea_block4)
        return self.x_fea

    def forward_gnn(self, zs):
        # gnn inp: n_q * n_way(n_s + 1) * f
        nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
        scores = self.gnn(nodes)
        # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
        scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0,
                                                                                                         2).contiguous().view(
            -1, self.n_way)
        return scores

    def predict(self, fea):
        z = self.fc(fea)
        z = z.view(self.n_way, -1, z.size(1))
        z_stack = [
            torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1,
                                                                                                            z.size(2))
            for i in range(self.n_query)]
        assert (z_stack[0].size(1) == self.n_way * (self.n_support + 1))
        # print('z_stack:', 'len:', len(z_stack), 'z_stack[0]:', z_stack[0].size())
        scores = self.forward_gnn(z_stack)
        return scores

    def feat_predict(self, x):
        x = x.cuda()
        # reshape the feature tensor: n_way * n_s + 15 * f
        assert (x.size(1) == self.n_support + 15)
        x = x.view(-1, *x.size()[2:])
        return self.predict(x)

    def get_classification_scores(self, z, classifier):
        z_norm = torch.norm(z, p=2, dim=1).unsqueeze(1).expand_as(z)
        z_normalized = z.div(z_norm + EPS)
        L_norm = torch.norm(classifier.weight.data, p=2, dim=1).unsqueeze(1).expand_as(classifier.weight.data)
        classifier.weight.data = classifier.weight.data.div(L_norm + EPS)
        cos_dist = classifier(z_normalized)
        cos_fac = 1.0
        scores = cos_fac * cos_dist
        return scores

    def forward_loss(self, x, y_cls, data_flag, standard_path=True, detach=False):
        x = x.cuda()
        x = x.view(-1, *x.size()[2:])
        if standard_path:
            x_fea = self.standard_path(x)
        else:
            x_fea = self.masked_path(x, detach=detach)
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = y_query.cuda()
        # todo fsl loss
        fsl_scores = self.predict(x_fea)
        fsl_loss = self.loss_fn(fsl_scores, y_query)

        # todo cls loss
        # for FC - global FC classifier
        if (data_flag == 'S'):
            cls_scores = self.get_classification_scores(x_fea, self.classifier_source)
            y_cls = y_cls.view(cls_scores.size()[0]).cuda()
            cls_loss = self.loss_fn(cls_scores, y_cls)
        elif (data_flag == 'A'):
            cls_scores = self.get_classification_scores(x_fea, self.classifier_target)
            y_cls = y_cls.view(cls_scores.size()[0]).cuda()
            cls_loss = self.loss_fn(cls_scores, y_cls)
        else:
            cls_loss = 0

        return fsl_scores, fsl_loss, cls_loss

    def reservoir_sampling(self, p, aug_rate):
        # k = torch.rand_like(p) ** (1 / p)
        _, indices = p.topk(int(p.shape[1] * aug_rate), dim=1)
        mask = torch.ones_like(p)
        mask[:, indices.squeeze(), :, :] = 0
        mask = mask.bool()
        return mask

    def train_loop(self, epoch, S_train_loader, A_train_loader, optimizer, total_it):

        def zero_grads():
            optimizer.zero_grad()

        def activation_sum(scores: torch.tensor):
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
            y_query = y_query.cuda()
            one_hot = F.one_hot(y_query, self.n_way)
            l = torch.sum(one_hot * scores.softmax(dim=1))
            return l

        print_freq = len(S_train_loader) // 10
        avg_loss = 0

        for ((i, (S_x, S_y_global)), (i, (A_x, A_y_global))) in zip(enumerate(S_train_loader),
                                                                    enumerate(A_train_loader)):
            self.n_query = S_x.size(1) - self.n_support
            if self.change_way:
                self.n_way = S_x.size(0)
            self.eval()
            A_ds_scores, A_ds_loss_fsl, A_ds_loss_cls = self.forward_loss(A_x, A_y_global, data_flag='A')
            l2 = [self.x_fea_block1, self.x_fea_block2,
                  self.x_fea_block3, self.x_fea_block4]
            tar_grad = torch.autograd.grad(A_ds_loss_fsl, l2, create_graph=False,
                                           allow_unused=True, retain_graph=True)
            tar_scores = [(tg * feat).detach().sum((0, 2, 3))  for tg, feat in zip(tar_grad, l2)]
            masks = [self.reservoir_sampling(a.view(1,-1,1,1),self.args.aug_rate) for a in tar_scores]
            self.train()
            self.MSA1 = masks[0]
            self.MSA2 = masks[1]
            self.MSA3 = masks[2]
            self.MSA4 = masks[3]

            S_ge_scores, S_ge_loss_fsl, S_ge_loss_cls = self.forward_loss(S_x, S_y_global, data_flag='S',
                                                                                      standard_path=True)
            self.MSA1 = ~masks[0]
            self.MSA2 = ~masks[1]
            self.MSA3 = ~masks[2]
            self.MSA4 = ~masks[3]
            A_ge_scores, A_ge_loss_fsl, A_ge_loss_cls = self.forward_loss(A_x, A_y_global, data_flag='A',
                                                                          standard_path=False)
            # todo random dropout channels which shared by all images
            lconsist = self.skl(A_ge_scores, A_ds_scores.detach())
            # STD: loss
            S_loss_ge = S_ge_loss_fsl + S_ge_loss_cls
            A_loss_ge = A_ge_loss_fsl + A_ge_loss_cls
            A_loss_ds = A_ds_loss_fsl+ A_ds_loss_cls

            loss = S_loss_ge + A_loss_ds + A_loss_ge + self.args.lconsist * lconsist
            zero_grads()
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if (i + 1) % print_freq == 0:
                print(
                    yellow_text(f'Epoch {epoch} ') + '| Batch {:d}/{:d} | Loss {:f}'.format(i + 1, len(S_train_loader),
                                                                                            avg_loss / float(i + 1)),
                    end='\n' if (i + 1) == len(S_train_loader) else '\r')

            if (total_it + 1) % 1 == 0 and self.tf_writer is not None:
                self.tf_writer.add_scalar(self.method + '/total_loss', loss.item(), total_it + 1)

            total_it += 1
        return total_it

    def test_loop(self, test_loader, record=None, prefix=''):
        loss = 0
        count = 0
        acc_all = []
        acc_all_mask = []

        iter_num = len(test_loader)
        for i, (x, _) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way = x.size(0)
            correct_this, correct_mask, count_this, loss_this = self.correct(x)
            acc_all.append(correct_this / count_this * 100)
            acc_all_mask.append(correct_mask / count_this * 100)
            loss += loss_this
            count += count_this

        acc_all_mask = np.asarray(acc_all_mask)
        acc_mask_mean = np.mean(acc_all_mask)
        acc_mask_std = np.std(acc_all_mask)
        print(f'--- {prefix} %d Loss = %.6f ---' % (iter_num, loss / count))
        print(f'--- {prefix} %d Mask Test Acc = %4.2f%% +- %4.2f%% ---' % (
            iter_num, acc_mask_mean, 1.96 * acc_mask_std / np.sqrt(iter_num)))
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print(f'--- {prefix} %d Test Acc = %4.2f%% +- %4.2f%% ---' % (
            iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        return acc_mean, acc_std/ np.sqrt(iter_num), acc_mask_mean, acc_mask_std, loss / count