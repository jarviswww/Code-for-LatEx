import argparse
import pickle
import time
import os
from model import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Tmall-2', help='dataset name: diginetica-2/Tmall-2/30music-2')
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.6, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=1, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,help='split the portion of training set as validation set')
# -------- LatEx --------- #
parser.add_argument('--intent_learning', default='average', help='cluster/average')
parser.add_argument('--temperature', type=float, default=0.1, help='the temperature coefficient of LatEx')
parser.add_argument('--slack', default=False, help='LatEx or Slack-LatEx')
parser.add_argument('--scale', type=float, default=0.2, help='the sacle of LatEx loss')
parser.add_argument('--threshold', type=int, default=8, help='the threshold of Slack-LatEx')

opt = parser.parse_args(args=[])
print(opt)


def main():
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    if opt.intent_learning == 'average':
        item_to_cat = pickle.load(open('datasets/' + opt.dataset + '/item_cate.txt', 'rb'))
        item_to_cat[0] = 0
        item_to_cat = renumber(item_to_cat)
        cat_to_item = {}
        for key, value in item_to_cat.items():
            cat_to_item.setdefault(value, []).append(key)
        item_to_cat = item_to_centroids(item_to_cat).cuda()
    else:
        item_to_cat = 0
        cat_to_item = 0

    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)

    if opt.dataset == 'retailrocket-2':
        n_node = 48990
    elif opt.dataset == 'diginetica-2':
        n_node = 43098
    elif opt.dataset == 'Tmall-2':
        n_node = 40728
    else:
        n_node = 0

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        train_test(model, opt, train_data, test_data, item_to_cat, cat_to_item, n_node)
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()