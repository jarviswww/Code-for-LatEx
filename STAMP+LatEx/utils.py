import networkx as nx
import numpy as np
from torch.utils.data import Dataset
import torch


def get_recall(pre, truth):
    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B,1) the truth value of test samples
    :return: recall(Float), the recall score
    """
    truths = truth.expand_as(pre)
    hits = (pre == truths).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (pre == truths).nonzero().size(0)
    recall = n_hits / truths.size(0)
    return recall


def get_mrr(pre, truth):
    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B, 1) real label
    :return: MRR(Float), the mrr score
    """
    targets = truth.view(-1, 1).expand_as(pre)
    # ranks of the targets, if it appears in your indices
    hits = (targets == pre).nonzero()
    if len(hits) == 0:
        return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    r_ranks = torch.reciprocal(ranks)  # reciprocal ranks
    mrr = torch.sum(r_ranks).data / targets.size(0)
    return mrr


def intraConv(cat_to_item, pre, pre_cate, truth_cate):

    D = torch.zeros_like(pre)
    score = 0
    for i in range(pre.size(0)):
        cat_item_num = len(cat_to_item[truth_cate[i].item()])
        indices = (pre_cate[i] == truth_cate[i].item()).nonzero()
        D[i, indices] = pre[i, indices]

        item_set = torch.unique(D[i], return_counts=True)[0]
        item_occur = torch.unique(D[i], return_counts=True)[1]
        mask = torch.where(item_occur == 1, 1, 0)
        unique_cat_item = item_set * mask
        unique_cat_item = torch.where(unique_cat_item == 0, 0, 1)
        num = torch.sum(unique_cat_item)
        score += num / cat_item_num

    return score / pre.shape[0]



def Conv(pre, n_node):

    score = 0
    item = []
    for i in range(pre.size(0)):
        item.append(pre[i].detach().cpu().numpy())
    item_num = len(np.unique(item))
    return item_num / n_node


def item_to_centroids(item_to_cat):
    sorted_keys = sorted(item_to_cat.keys())
    unique_keys_count = len(sorted_keys)
    matrix = np.zeros((unique_keys_count))
    for i, key in enumerate(sorted_keys):
        matrix[i] = item_to_cat[key]
    return torch.tensor(matrix).long()


def renumber(original_dict):

    sorted_values = np.unique(sorted(original_dict.values(), reverse=True))
    value_indices = {value: index for index, value in enumerate(sorted_values)}
    new_dict = {key: value_indices[value] for key, value in original_dict.items()}

    return new_dict


def data_masks(all_usr_pois, is_train, max_len_limit=None):
    print(len(all_usr_pois), "************")

    print(len(all_usr_pois))
    us_lens = [len(upois) for upois in all_usr_pois]
    if max_len_limit is None:
        len_max = max(us_lens)
    else:
        len_max = max_len_limit
    print(len_max)
    us_pois = [upois[-1 * len_max:] + [0] * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1 if i != 0 else 0 for i in upois] for upois in us_pois]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set[0], train_set[1]
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data(Dataset):
    def __init__(self, data, is_train):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, is_train, 20)
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.max_n_node = max([len(np.unique(i)) for i in inputs])

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[
            index]

        return [torch.tensor(u_input).long(), torch.tensor(mask).long(), torch.tensor(target).long()]

    def __len__(self):
        return self.length
