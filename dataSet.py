import os
from torch.utils.data import DataLoader, Dataset
from utils import *


def batch_matching_matrix(labels):
    r"""

    :param labels: list of matching matrix, labels = [M1, M2, ..., Mi]
    :return: batch_matching_matrix, shape of BM is (sum of ni) * (sum of mi)
             mask, location for 1 represnt the location require masking, mask is (sum of ni) * (sum of mi)
    """
    n = [0]
    m = [0]
    for i in labels:
        n.append(i.shape[0])
        m.append(i.shape[1])
    bmm = torch.zeros(sum(n), sum(m), dtype=torch.float32)
    mask = torch.ones(sum(n), sum(m), dtype=torch.float32)
    n = list(np.cumsum(n))
    m = list(np.cumsum(m))
    for i in np.arange(len(n) - 1):
        bmm[n[i]:n[i + 1], m[i]:m[i + 1]] = labels[i]
        mask[n[i]:n[i + 1], m[i]:m[i + 1]] = 0
    return bmm, mask


class dgraph(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.graph_pairs = os.listdir(self.root_dir)

    def __getitem__(self, index):
        graph_pair_index = self.graph_pairs[index]
        graph_pair_path = os.path.join(self.root_dir, graph_pair_index)
        graph_pair, label_dict = load_graphs(graph_pair_path)
        graph_da = graph_pair[0]
        graph_q = graph_pair[1]
        label = label_dict['glabel']
        return dgl.add_self_loop(graph_da), dgl.add_self_loop(graph_q), label

    def __len__(self):
        return len(self.graph_pairs)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    g1, g2, labels = map(list, zip(*samples))
    bg1 = dgl.batch(g1)
    bg2 = dgl.batch(g2)
    batchmatchingmatrix, mask = batch_matching_matrix(labels)
    return bg1, bg2, torch.tensor(batchmatchingmatrix, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


if __name__=='__main__':
    device = torch.device('cuda:0')
    path_train = './SYNTHETIC/'
    # path_test = './opendata/test/'
    batch_size = 2
    d_train = dgraph(root_dir=path_train)
    data_loader_train = DataLoader(d_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
    end = []
    for j, (bbg_da, bbg_q, b_mm, b_mask) in enumerate(data_loader_train):
        print(j)
        bbg_q = bbg_q.to(device)
        print(bbg_q.batch_num_nodes())
        # print(bbg_q.num_nodes())
        print(b_mm.shape)
        print(len(np.where(b_mm == 1)[1]))
        print(b_mm.sum(-1).sum())
        break










