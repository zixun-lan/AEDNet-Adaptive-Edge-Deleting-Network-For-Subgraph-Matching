from dgl.data.utils import save_graphs, get_download_dir, load_graphs
import torch
import dgl
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher, DiGraphMatcher
import networkx.algorithms.isomorphism as iso
import random


def 增一点(已选, datu):
    可选 = set()
    for i in 已选:
        s_n = set(datu.predecessors(i).numpy())
        d_n = set(datu.successors(i).numpy())
        i_n = s_n | d_n
        可选 = 可选 | i_n
    可选 = 可选 - set(已选)
    可选 = list(可选)
    # print(可选)
    一点 = random.choice(list(可选))
    return 一点


def sg(seed, datu, size):
    已选 = [int(seed)]
    while len(已选) < size:
        N = 增一点(已选=已选, datu=datu)
        已选.append(int(N))
    已选 = set(已选)
    已选 = list(已选)
    已选 = sorted(已选)
    if len(已选) != size:
        print('wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    sub_g = dgl.node_subgraph(datu, 已选)
    return sub_g, np.array(已选)


def to_m(label, q_size, g_size):  # label 是 np1d.array
    m = np.zeros([q_size, g_size])
    for i, j in enumerate(label):
        m[i][j] = 1.
    return m


def test_data():
    nm = iso.numerical_node_match('x', 1.0)
    glist, label_dict = load_graphs("./synthetic/109.bin")
    g = glist[0]
    q = glist[1]
    label = label_dict['glabel']
    a = torch.arange(100, dtype=torch.float32).reshape(100, 1)
    print(g)
    print(q)
    print(label)
    nx_da = dgl.to_networkx(g, node_attrs=['x'])
    nx_q = dgl.to_networkx(q, node_attrs=['x'])
    mm = DiGraphMatcher(G1=nx_da, G2=nx_q, node_match=nm)
    aa = list(mm.subgraph_isomorphisms_iter())
    print(aa)
    c = torch.mm(label, a)
    print(c)


def new_label(aa, q_size, g_size):  # aa是list，[dict1, dict2, ... ,]
    newlabel = np.zeros([q_size, g_size])
    for i in aa:  # i 是 dict，{68: 0, 592: 1, 1784: 3, 1306: 2, 3990: 4, 8577: 5, 8574: 6}
        ii = list(i.items())  # ii是list, [(68, 0), (592, 1), (1784, 3), (1306, 2), (3990, 4), (8577, 5), (8574, 6)]
        for j in ii:  # j是元组， (68, 0)
            newlabel[j[1]][j[0]] = 1
    return newlabel


def process_batch_matching_matrix(bbg_da, b_mm):
    match_nodes = sorted(list(set(list(np.where(b_mm.numpy() == 1)[1]))))
    nodeSubgarph = dgl.node_subgraph(bbg_da, match_nodes)
    PEid = sorted(list(set(list(nodeSubgarph.edata[dgl.EID].numpy()))))
    ALLid = sorted(list(set(list(bbg_da.in_edges(match_nodes, form='eid').numpy()))))
    DEid = set(ALLid) - set(PEid)
    DEid = sorted(list(DEid))
    return match_nodes, PEid, DEid


def inx_bg_num_nodes(bg_num_nodes):  # bg_num_nodes 是 torch.tensor 1d
    bg_num_nodes = bg_num_nodes.cpu().numpy()
    end = []
    for i, j in enumerate(bg_num_nodes):
        end.append(np.full(j, i))
    end = np.concatenate(end)
    return end


def total_loss(lambda_step_T, lambda_step_T_DE, lambda_DE, mm_step_T, List_negative_DE, List_MM, b_mm_label, criterion, device):
    # criterion = torch.nn.MSELoss().to(device)
    # 计算0到t-1的loss
        # 计算MM_loss
    predicted_MM_t_1 = torch.cat(List_MM[0:-1], dim=0)  # predicted_MM_t_1 = (n*(layers-1)) x m
    label_MM_for_t_1 = b_mm_label.repeat(len(List_MM[0:-1]), 1)  # label_MM_for_t_1 = (n*(layers-1)) x m
    inv_label_MM_for_t_1 = (label_MM_for_t_1 == 0).to(torch.float32)
    tmp_1_MM_t_1 = (predicted_MM_t_1 * label_MM_for_t_1).sum(-1).reshape(len(List_MM[0:-1]) * b_mm_label.shape[0], 1)
    tmp_0_MM_t_1 = (predicted_MM_t_1 * inv_label_MM_for_t_1).sum(-1).reshape(len(List_MM[0:-1]) * b_mm_label.shape[0], 1)
    negative_tmp_MM_t_1 = - (tmp_1_MM_t_1 - tmp_0_MM_t_1)  # -(1-0) = (n*(layers-1)) x 1
    negative_tmp_MM_t_1 = 1 + negative_tmp_MM_t_1  # 1 + (-(1-0)) = (n*(layers-1)) x 1
    label_for_MM_t_1 = torch.zeros(len(List_MM[0:-1]) * b_mm_label.shape[0], 1, dtype=torch.float32).to(device)  # (n*(layers-1)) x 1
    loss_for_MM_t_1 = criterion(negative_tmp_MM_t_1, label_for_MM_t_1)
        # 计算DE_loss
    predicted_negative_DE_t_1 = torch.cat(List_negative_DE[0:-1], dim=0)  # predicted_negative_DE_t_1 = (layers-1) x 1 # -(1-0)
    predicted_negative_DE_t_1 = 1 + predicted_negative_DE_t_1  # 1+(-(1-0)) = (layers-1) x 1
    label_for_DE_t_1 = torch.zeros(len(List_negative_DE[0:-1]), 1, dtype=torch.float32).to(device)  # (layers-1) x 1
    loss_for_DE_t_1 = criterion(predicted_negative_DE_t_1, label_for_DE_t_1)
        # t-1的总loss
    loss_for_t_1 = lambda_DE * loss_for_DE_t_1 + (1 - lambda_DE) * loss_for_MM_t_1
    # 计算T的loss
        # 计算MM_loss
    inv_label_MM_for_T = (b_mm_label == 0).to(torch.float32)  # inv_label_MM_for_T = n x m
    tmp_1_MM_T = (mm_step_T * b_mm_label).sum(-1).reshape(b_mm_label.shape[0], 1)  # tmp_1_MM_T = n x 1
    tmp_0_MM_T = (mm_step_T * inv_label_MM_for_T).sum(-1).reshape(b_mm_label.shape[0], 1)  # tmp_0_MM_T = n x 1
    negative_tmp_MM_T = - (tmp_1_MM_T - tmp_0_MM_T)  # negative_tmp_MM_T = n x 1
    negative_tmp_MM_T = 1 + negative_tmp_MM_T  # 1+(-(1-0)) = n x 1
    label_for_MM_T = torch.zeros(b_mm_label.shape[0], 1, dtype=torch.float32).to(device)  # n x 1
    loss_for_MM_T = criterion(negative_tmp_MM_T, label_for_MM_T)
        # 计算DE_loss
    predicted_negative_DE_T = List_negative_DE[-1]  # -(1-0) = 1 x 1
    predicted_negative_DE_T = 1 + predicted_negative_DE_T  # 1+(-(1-0)) = 1 x 1
    loss_for_DE_T = criterion(predicted_negative_DE_T, torch.zeros(1, 1, dtype=torch.float32).to(device))
        # 计算T的总loss
    loss_for_T = lambda_step_T_DE * loss_for_DE_T + (1 - lambda_step_T_DE) * loss_for_MM_T
    # total loss
    loss_total = lambda_step_T * loss_for_T + (1 - lambda_step_T) * loss_for_t_1
    return loss_total


def metric_f1(preticted_MM, b_mm_label):
    preticted_MM = preticted_MM.cpu().detach()
    b_mm_label = b_mm_label.cpu().numpy()
    inx = torch.max(preticted_MM, dim=1)[1].numpy()
    row = np.arange(b_mm_label.shape[0])
    value = b_mm_label[row, inx]
    f1 = np.sum(value) / len(value)
    return f1


























