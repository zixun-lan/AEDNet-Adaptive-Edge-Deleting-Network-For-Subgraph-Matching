from dgl.data import TUDataset
from utils import *
import networkx as nx


def creat_dataset(dataset_name, min_a, max_b, num_of_example, save_path, label_or_attr):
    r"""
    dataset_name 字符串, min_a 子图最小点, max_b 子图最大点, num_of_example 生成样本数, save_path 字符串, label_or_attr 字符串
    """
    nm = iso.numerical_node_match('x', 1.0)
    data = TUDataset(dataset_name)
    i = 0
    while i < num_of_example:
        # print(i)
        for j in np.arange(len(data)):
            print('第几个', i, 'wwwwwwwwwwwwwwwwwwwww')
            g, label = data[j]
            nxg = dgl.to_networkx(g)
            if nx.is_connected(nxg.to_undirected()) is not True:
                continue
            da_dgl = dgl.from_networkx(nxg)
            da_dgl.ndata['x'] = g.ndata[label_or_attr]
            q_size = random.randint(min_a, max_b)
            print(q_size)

            kaiguan = True
            while kaiguan:
                seed = random.randint(0, da_dgl.num_nodes() - 1)
                print('seed: ', seed)
                q_dgl, lllable = sg(seed=seed, datu=da_dgl, size=q_size)
                nx_q = dgl.to_networkx(q_dgl)
                if nx.is_connected(nx_q.to_undirected()) is not True:  # 检查生成的子图 是否连通
                    print('错误错误！！！！！！！生成子图不连通')

                # 检查是否唯一子图
                nx_da = dgl.to_networkx(da_dgl, node_attrs=['x'])
                nx_q = dgl.to_networkx(q_dgl, node_attrs=['x'])
                mm = DiGraphMatcher(G1=nx_da, G2=nx_q, node_match=nm)
                aa = list(mm.subgraph_isomorphisms_iter())
                print(aa)
                if len(aa) == 1:
                    da_dgl.ndata['x'] = torch.tensor(da_dgl.ndata['x'], dtype=torch.float32)
                    q_dgl.ndata['x'] = torch.tensor(q_dgl.ndata['x'], dtype=torch.float32)
                    matching_matrix = to_m(label=lllable, q_size=q_dgl.num_nodes(), g_size=da_dgl.num_nodes())
                    graph_labels = {'glabel': torch.tensor(matching_matrix, dtype=torch.float32)}
                    path = save_path + str(i) + '.bin'
                    save_graphs(path, [da_dgl, q_dgl], graph_labels)
                    i = i + 1
                    kaiguan = False
            if i == num_of_example:
                break


def general_creat_dataset(dataset_name, min_a, max_b, num_of_example, save_path, label_or_attr):
    r"""
    dataset_name 字符串, min_a 子图最小点, max_b 子图最大点, num_of_example 生成样本数, save_path 字符串, label_or_attr 字符串
    """
    nm = iso.numerical_node_match('x', 1.0)
    data = TUDataset(dataset_name)
    i = 0
    while i < num_of_example:
        # print(i)
        for j in np.arange(len(data)):
            print('第几个', i, 'wwwwwwwwwwwwwwwwwwwww')
            g, label = data[j]
            nxg = dgl.to_networkx(g)
            if nx.is_connected(nxg.to_undirected()) is not True:
                continue
            da_dgl = dgl.from_networkx(nxg)
            da_dgl.ndata['x'] = g.ndata[label_or_attr]
            q_size = random.randint(min_a, max_b)
            print(q_size)

            seed = random.randint(0, da_dgl.num_nodes() - 1)
            print('seed: ', seed)
            q_dgl, lllable = sg(seed=seed, datu=da_dgl, size=q_size)
            nx_q = dgl.to_networkx(q_dgl)
            if nx.is_connected(nx_q.to_undirected()) is not True:  # 检查生成的子图 是否连通
                print('错误错误！！！！！！！生成子图不连通')

            # 检查是否唯一子图
            nx_da = dgl.to_networkx(da_dgl, node_attrs=['x'])
            nx_q = dgl.to_networkx(q_dgl, node_attrs=['x'])
            mm = DiGraphMatcher(G1=nx_da, G2=nx_q, node_match=nm)
            aa = list(mm.subgraph_isomorphisms_iter())
            print(aa)
            print('有多少个子图：', len(aa))

            matching_matrix = new_label(aa=aa, q_size=q_size, g_size=da_dgl.num_nodes())
            graph_labels = {'glabel': torch.tensor(matching_matrix, dtype=torch.float32)}
            path = save_path + str(i) + '.bin'
            save_graphs(path, [da_dgl, q_dgl], graph_labels)
            i = i + 1
            if i == num_of_example:
                break


dataset_name = 'SYNTHETIC'
min_a = 25
max_b = 40
num_of_example = 10000
save_path = './SYNTHETIC/'
label_or_attr = 'node_labels'
general_creat_dataset(dataset_name, min_a, max_b, num_of_example, save_path, label_or_attr)
