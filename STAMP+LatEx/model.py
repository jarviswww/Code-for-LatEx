import datetime
import math
import numpy as np
import torch
import torch.nn.functional as F
from utils import *
from torch import nn
from torch.nn import Module, Parameter
from tqdm import tqdm


class ContrastiveLoss(nn.Module):
    def __init__(self, threshold, batch_size, slack_LatEx, temperature):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.slack_LatEx = slack_LatEx
        self.threshold = threshold
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j, cluster_embeddings, target_clsuter):
        if self.slack_LatEx == False:
            SIZE = emb_i.shape[0]
            z_i = F.normalize(emb_i, dim=1)
            z_j = F.normalize(emb_j, dim=1)
        else:
            SIZE = emb_i.shape[0]

            expand_emb_j = emb_j.unsqueeze(1).expand(emb_j.shape[0], cluster_embeddings.shape[0], emb_j.shape[1])
            expand_cluster_embeddings = cluster_embeddings.unsqueeze(0).expand(emb_j.shape[0],
                                                                               cluster_embeddings.shape[0],
                                                                               emb_j.shape[1])

            cluster_similarity_matrix = torch.cosine_similarity(expand_emb_j, expand_cluster_embeddings, -1)
            rank_cluster_similarity_matrix = torch.argsort(cluster_similarity_matrix)[:, -self.threshold:]
            slack_positives = cluster_embeddings[rank_cluster_similarity_matrix]
            z_j = torch.sum(slack_positives, dim=1)

            z_i = F.normalize(emb_i, dim=1)
            z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.mm(representations, representations.t().contiguous())
        sim_ij = torch.diag(similarity_matrix, SIZE)
        sim_ji = torch.diag(similarity_matrix, -SIZE)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        negatives_mask = (~torch.eye(SIZE * 2, SIZE * 2, dtype=bool)).cuda().float()
        negative_sample_mask = self.sample_mask(target_clsuter)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
        denominator = negative_sample_mask * denominator  
        loss_partial = -torch.log(nominator / (torch.sum(denominator, dim=1) + 1e-7))
        loss = torch.sum(loss_partial) / (2 * SIZE)
        return loss

    def sample_mask(self, targets):
        targets = targets.cpu().numpy()
        targets = np.concatenate([targets, targets])

        cl_dict = {}
        for i, target in enumerate(targets):
            cl_dict.setdefault(target, []).append(i)
        mask = np.ones((len(targets), len(targets)))
        for i, target in enumerate(targets):
            for j in cl_dict[target]:
                if abs(j - i) != len(targets) / 2:  
                    mask[i][j] = 0
        return torch.Tensor(mask).cuda().float()


class LatEx(nn.Module):
    def __init__(self, dim, threshold, batch_size, slack_LatEx, temperature=0.5):
        super(LatEx, self).__init__()
        self.ContrastiveLoss = ContrastiveLoss(threshold, batch_size, slack_LatEx, temperature)

    def forward(self, session_rep, item_2cluster, item_centroids, target):
        session_rep = session_rep
        item_centroids = item_centroids

        target_cluster = item_2cluster[target]
        target_cluster_embeddings = item_centroids[target_cluster]

        LatEx_loss = self.ContrastiveLoss(session_rep, target_cluster_embeddings, item_centroids, target_cluster)
        return LatEx_loss


class SessionGraph(Module):
    def __init__(self, opt, n_node, cat_to_item, item_to_cat):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.threshold = opt.threshold
        self.slack = opt.slack
        self.temperature = opt.temperature
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.cat_to_item = cat_to_item
        self.item_to_cat = item_to_cat
        self.intent_learning = opt.intent_learning
        self.scale = opt.scale

        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_four = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_ht = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_hs = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.ln1 = torch.nn.LayerNorm([self.hidden_size, ], elementwise_affine=False)

        self.LatEx = LatEx(opt.hiddenSize, self.threshold, self.batch_size, self.slack, self.temperature)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def compute_scores(self, hidden, mask, t_score, targets):
        cat_to_item = self.cat_to_item
        item_to_cat = self.item_to_cat

        # ----------- intent learning (Explicit)------- #
        if self.intent_learning == 'average':
            cat_to_item = {k: sorted(v) for k, v in sorted(cat_to_item.items())}
            l1 = cat_to_item.values()
            l2 = max(map(len, cat_to_item.values()))
            new = []
            count = []
            for i in l1:
                q1 = list(i) + [0] * (l2 - len(i))
                q2 = [1] * len(i) + [0] * (l2 - len(i))
                new.append(q1)
                count.append(q2)
            new = torch.tensor(new).cuda()
            count = torch.tensor(count).cuda()
            new_embeds = self.embedding(new) * count.unsqueeze(-1).repeat(1, 1, self.embedding.weight.shape[-1])

            l = torch.sum(count, dim=-1).unsqueeze(-1).repeat(1, new_embeds.shape[-1])
            item_centroids = torch.sum(new_embeds, dim=1) / torch.sum(count, dim=-1).unsqueeze(-1).repeat(1,
                                                                                                          new_embeds.shape[
                                                                                                              -1])
            self.item_centroids = item_centroids
            self.item_2cluster = item_to_cat
            self.cat_to_item = cat_to_item

        # ---------------- SBR model ----------------- #
        xt = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask,1) - 1]  # batch_size x latent_size
        ht = torch.tanh(self.linear_ht(xt))  # batch_size x latent_size

        q1 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        q2 = self.linear_one(xt).view(xt.shape[0], 1, xt.shape[1])  # batch_size x 1 x latent_size
        ms = torch.sum(hidden * mask.view(mask.shape[0], -1, 1).float(), 1) / torch.sum(mask.float(), -1).unsqueeze(-1)
        q3 = self.linear_three(ms).view(xt.shape[0], 1, xt.shape[1])  # batch_size x 1 x latent_size
        alpha = self.linear_four(torch.sigmoid(q1 + q2 + q3))  # batch_size x seq_length x 1
        ma = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(),1) + ms  # batch_size x latent_size
        hs = torch.tanh(self.linear_hs(ma))  # batch_size x latent_size
        a = self.linear_transform(torch.cat([hs, ht], 1))  # batch_size x latent_size

        LatEx_loss = self.LatEx(a, self.item_2cluster, self.item_centroids, targets)

        v = self.embedding.weight[1:]  # n_nodes x latent_size
        scores = torch.matmul(a, v.transpose(1, 0))

        return scores, 0, 0, ht[0], LatEx_loss * self.scale

    def e_step(self):
        items_embedding = self.embedding.weight.detach().cpu().numpy()
        self.item_centroids, self.item_2cluster = self.run_kmeans(items_embedding[:])

    def run_kmeans(self, x):
        kmeans = faiss.Kmeans(d=x.shape[-1], k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        self.cluster_cents = cluster_cents
        _, I = kmeans.index.search(x, 1)
        self.items_cents = I

        cluster_dict = {}
        for i, cluster_id in enumerate(I.flatten()):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(i)
        self.cat_to_item = {k: sorted(v) for k, v in sorted(cluster_dict.items())}

        centroids = torch.Tensor(cluster_cents).cuda()
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster

    def forward(self, inputs):
        hidden = self.embedding(inputs)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, t_score):
    alias_inputs, mask, targets = data
    alias_inputs = trans_to_cuda(alias_inputs)
    mask = trans_to_cuda(mask)
    seq_hidden = model(alias_inputs)
    scores, temp_0, temp_1, temp_2, LatEx_loss = model.compute_scores(seq_hidden, mask, t_score, targets)
    return targets, scores, temp_0, temp_1, temp_2, LatEx_loss


def train_test(model, opt, train_data, test_data, t_score, n_node):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_size=model.batch_size,
                                               ## , num_workers=4
                                               shuffle=True, pin_memory=True)

    for j, data in tqdm(enumerate(train_loader), total=train_loader.__len__()):
        model.optimizer.zero_grad()

        # ----------- intent learning (Latent)------- #
        if opt.intent_learning == 'cluster' and step % 5 == 0:
            model.e_step()

        targets, scores, temp_0, temp_1, temp_2, LatEx_loss = forward(model, data, t_score)
        targets = trans_to_cuda(targets)
        if float(temp_0) > 0.1:
            loss = model.loss_function(scores,targets - 1) + temp_0
        else:
            loss = model.loss_function(scores, targets - 1)
        loss = loss + LatEx_loss

        if j % 500 == 0:
            print("Loss:{}".format(loss))
            print("LatEx_Loss:{}".format(LatEx_loss))

        loss.backward()
        model.optimizer.step()
        total_loss += loss

    print('\tLoss:\t%.3f' % total_loss)
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit_20, mrr_20, hit_10, mrr_10 = [], [], [], []
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=1, batch_size=model.batch_size,
                                              shuffle=True, pin_memory=True)
    flag = 0
    y_pre_all = torch.LongTensor().cuda()
    y_pre_all_10 = torch.LongTensor().cuda()
    test_y = torch.LongTensor().cuda()

    for data in tqdm(test_loader):
        targets, scores, temp_0, temp_1, temp_2, LatEx_loss = forward(model, data, t_score)
        targets = targets.numpy()
        y_pre = scores.topk(20)[1]
        targets = torch.Tensor(targets).long().cuda()

        test_y = torch.cat((test_y, targets), 0)
        y_pre_all = torch.cat((y_pre_all, y_pre), 0)
        y_pre_all_10 = torch.cat((y_pre_all_10, y_pre[:, :10]), 0)

    pre_cat = model.item_2cluster[y_pre_all + 1]
    pre_cat_10 = model.item_2cluster[y_pre_all_10 + 1]
    test_cat = model.item_2cluster[test_y]
    cat_to_item = model.cat_to_item

    Int_10 = IntMetric(cat_to_item, y_pre_all_10, pre_cat_10, test_cat.cuda().unsqueeze(1))
    Int_20 = IntMetric(cat_to_item, y_pre_all, pre_cat, test_cat.cuda().unsqueeze(1))
    Cov = Coverage(y_pre_all, n_node)

    recall = get_recall(y_pre_all, test_y.long().unsqueeze(1) - 1)
    recall_10 = get_recall(y_pre_all_10, test_y.unsqueeze(1) - 1)
    mrr = get_mrr(y_pre_all, test_y.long().unsqueeze(1) - 1)
    mrr_10 = get_mrr(y_pre_all_10, test_y.unsqueeze(1) - 1)


    print("Coverage@20: " + "%.4f" % Cov)
    print("Int@20: " + "%.4f" % Int_20 + "  Int@10: " + "%.4f" % Int_10)
    print("Recall@20: " + "%.4f" % recall + "  Recall@10: " + "%.4f" % recall_10)
    print("MRR@20:" + "%.4f" % mrr.tolist() + "  MRR@10:" + "%.4f" % mrr_10.tolist())
