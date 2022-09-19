import torch.nn.functional as F
from utils import *
from dgl.nn.pytorch.glob import GlobalAttentionPooling
import math
from dgl.ops import edge_softmax
import torch.nn as nn
import torch as th
import dgl.function as fn


class project_layers(torch.nn.Module):
    def __init__(self, categoricalTo0_or_continuousTo1, input_dimension, h_dimension, p_drop=0, minmaxlist=None,
                 batchNorm=True, normAffine=False):
        super(project_layers, self).__init__()
        self.categoricalTo0_or_continuousTo1 = categoricalTo0_or_continuousTo1
        self.batchNorm = batchNorm
        if categoricalTo0_or_continuousTo1 == 0:
            self.project = torch.nn.Embedding(minmaxlist[1] + 1, h_dimension)
            self.dp = torch.nn.Dropout(p=p_drop)
        elif categoricalTo0_or_continuousTo1 == 1:
            self.project = torch.nn.Linear(input_dimension, h_dimension)
            self.dp = torch.nn.Dropout(p=p_drop)
            if batchNorm:
                self.bn = torch.nn.BatchNorm1d(h_dimension, affine=normAffine)

    def forward(self, x):
        if self.categoricalTo0_or_continuousTo1 == 0:
            x = x.reshape(-1).to(torch.long)
            y = self.project(x)
            y = self.dp(y)
        elif self.categoricalTo0_or_continuousTo1 == 1:
            y = self.project(x)
            if self.batchNorm:
                y = self.bn(y)
            y = torch.nn.functional.elu(y)
            y = self.dp(y)
        return y


class MatchingMatrixNormalization(torch.nn.Module):
    def __init__(self, modefornorm_0forSoftmax_1forTau_2forMean, dim=-1):
        super(MatchingMatrixNormalization, self).__init__()
        self.dim = dim
        self.modefornorm_0forSoftmax_1forTau_2forMean = modefornorm_0forSoftmax_1forTau_2forMean
        if modefornorm_0forSoftmax_1forTau_2forMean == 1:
            self.reset_parameters()

    def reset_parameters(self):
        self.w = nn.Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.w, 0)

    def forward(self, matchingMatrix, mask):
        inverse_mask = (mask == 0).to(torch.float32)
        if self.modefornorm_0forSoftmax_1forTau_2forMean == 0:
            matchingMatrix = matchingMatrix * inverse_mask
            matchingMatrix = matchingMatrix + (mask * -1e9)
            matchingMatrix = torch.nn.functional.softmax(matchingMatrix, dim=self.dim)
        elif self.modefornorm_0forSoftmax_1forTau_2forMean == 1:
            matchingMatrix = matchingMatrix * inverse_mask
            matchingMatrix = matchingMatrix / (torch.sigmoid(self.w))
            matchingMatrix = matchingMatrix + (mask * -1e9)
            matchingMatrix = torch.nn.functional.softmax(matchingMatrix, dim=self.dim)
        elif self.modefornorm_0forSoftmax_1forTau_2forMean == 2:
            matchingMatrix = matchingMatrix * inverse_mask
            denominator = matchingMatrix.sum(self.dim).reshape(-1, 1)
            matchingMatrix = matchingMatrix / denominator
        return matchingMatrix


class simlarity(torch.nn.Module):
    def __init__(self, modeforsim_0forE_1forDot_2forleakyrelu, h_dimension, negative_slope=0.2, tauForE=1):
        super(simlarity, self).__init__()
        self.h_dimension = h_dimension
        self.h_dimension_sqrt = h_dimension ** 0.5
        self.modeforsim_0forE_1forDot_2forleakyrelu = modeforsim_0forE_1forDot_2forleakyrelu
        if modeforsim_0forE_1forDot_2forleakyrelu == 0:
            self.tauForE = tauForE
            self.reset_parameters()
        elif modeforsim_0forE_1forDot_2forleakyrelu == 2:
            self.leaky_relu = nn.LeakyReLU(negative_slope)

    def reset_parameters(self):
        self.w = nn.Parameter(torch.FloatTensor(self.h_dimension, self.h_dimension))
        nn.init.xavier_uniform_(self.w)

    def forward(self, h_da, h_q):
        if self.modeforsim_0forE_1forDot_2forleakyrelu == 0:
            mm = torch.mm(h_q, self.w)
            mm = torch.mm(mm, h_da.T)
            mm = torch.exp(mm / self.tauForE)
        elif self.modeforsim_0forE_1forDot_2forleakyrelu == 1:
            mm = torch.mm(h_q, h_da.T)
            mm = mm / self.h_dimension_sqrt
        elif self.modeforsim_0forE_1forDot_2forleakyrelu == 2:
            mm = torch.mm(h_q, h_da.T)
            mm = mm / self.h_dimension_sqrt
            mm = self.leaky_relu(mm)
        return mm


class MatchingMatrix(torch.nn.Module):
    def __init__(self, modeforsim_0forE_1forDot_2forleakyrelu, modefornorm_0forSoftmax_1forTau_2forMean, h_dimension,
                 negative_slope=0.2, tauForE=1):
        super(MatchingMatrix, self).__init__()
        self.normalization_layer = MatchingMatrixNormalization(
            modefornorm_0forSoftmax_1forTau_2forMean=modefornorm_0forSoftmax_1forTau_2forMean)
        self.simlarity_layer = simlarity(modeforsim_0forE_1forDot_2forleakyrelu=modeforsim_0forE_1forDot_2forleakyrelu,
                                         h_dimension=h_dimension, negative_slope=negative_slope, tauForE=tauForE)

    def forward(self, h_da, h_q, mask):
        mm = self.simlarity_layer(h_da=h_da, h_q=h_q)
        mm = self.normalization_layer(mm, mask)
        return mm


class fully_neural_network(torch.nn.Module):
    def __init__(self, inp_dimension, mid_imension, oup_dimension, batchNorm=True, normAffine=False, p_drop=0):
        super(fully_neural_network, self).__init__()
        self.batchNorm = batchNorm
        if batchNorm:
            self.bn1 = torch.nn.BatchNorm1d(mid_imension, affine=normAffine)
            self.bn2 = torch.nn.BatchNorm1d(oup_dimension, affine=normAffine)
        self.L1 = nn.Linear(inp_dimension, mid_imension)
        self.L2 = nn.Linear(mid_imension, oup_dimension)
        self.dp = torch.nn.Dropout(p=p_drop)

    def forward(self, x):
        y = self.L1(x)
        if self.batchNorm:
            y = self.bn1(y)
        y = torch.nn.functional.elu(y)
        y = self.dp(y)
        y = self.L2(y)
        if self.batchNorm:
            y = self.bn2(y)
        y = torch.nn.functional.elu(y)
        y = self.dp(y)
        return y


class ATTpolling_for_3Dtensor(torch.nn.Module):
    def __init__(self, h_dimension):
        super(ATTpolling_for_3Dtensor, self).__init__()
        self.pooling1 = GlobalAttentionPooling(gate_nn=nn.Linear(h_dimension, 1))
        self.pooling2 = GlobalAttentionPooling(gate_nn=nn.Linear(h_dimension, 1))

    def forward(self, Q, G, M1, M2):  # Q 1 x d or b x d
        k_graph_level_M1 = self.pooling1(G, M1)  # batch_size x d
        k_graph_level_M2 = self.pooling2(G, M2)  # batch_size x d
        s1 = (Q * k_graph_level_M1).sum(dim=-1).unsqueeze(-1)  # batch_size x 1
        s2 = (Q * k_graph_level_M2).sum(dim=-1).unsqueeze(-1)  # batch_size x 1
        att = torch.cat([s1, s2], dim=-1)  # batch_size x 2
        att = torch.nn.functional.softmax(att, dim=-1)  # batch_size x 2
        np_inx = inx_bg_num_nodes(bg_num_nodes=G.batch_num_nodes())  # np.array  1d
        att = att[np_inx, :]  # N x 2
        end = M1 * att[:, 0].unsqueeze(-1) + M2 * att[:, 1].unsqueeze(-1)  # (N x d * N x 1) + (N x d * N x 1)
        return end


class CPADE_GAT_layer(torch.nn.Module):
    def __init__(self, h_dimension, num_head, residual=True, leaky_relu_GAT_rate=2, batchNorm=True, normAffine=False,
                 p_drop=0, drop_edge_p=0):
        super(CPADE_GAT_layer, self).__init__()
        self.h_dimension = h_dimension
        self.num_head = num_head
        self.each_head = math.ceil(h_dimension / num_head)
        self.Q_from_q = fully_neural_network(inp_dimension=h_dimension, mid_imension=2 * h_dimension,
                                             oup_dimension=2 * num_head * self.each_head, batchNorm=batchNorm,
                                             normAffine=normAffine, p_drop=p_drop)
        #  Q_from_q  用来将 bg_q 池化后的 b个graph-level embedding 转化为 b个对应的 全层的 Q  （b x (2*num_head*each_head)）
        self.L = nn.Linear(h_dimension, num_head * self.each_head)
        #  L 全层共享 将原始的节点embedding，转变为多头所对应的节点embedding。全层共享的目的：相同的原始节点embedding将被转化为相同的多头所对应的节点embedding
        self.pooling_q = GlobalAttentionPooling(
            gate_nn=nn.Linear(h_dimension, 1))  # 将bg_q转为 b个graph-level embedding  出来后(b x d)
        self.att_comb_initAndT = ATTpolling_for_3Dtensor(h_dimension=h_dimension)  # 将 h_t 和 h0 加权平均。 出来后还是 n x d
        self.Q_for_comb = nn.Parameter(th.FloatTensor(size=(1, h_dimension)))  # Q_for_comb 用来合并h_t 和 h0 的Q，全层共享
        self.end_FNN = fully_neural_network(inp_dimension=num_head * self.each_head, mid_imension=2 * h_dimension,
                                            oup_dimension=h_dimension, batchNorm=batchNorm, normAffine=normAffine,
                                            p_drop=p_drop)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_GAT_rate)
        self.drop_edge_da = torch.nn.Dropout(p=drop_edge_p)
        self.drop_edge_q = torch.nn.Dropout(p=drop_edge_p)
        self.residual = residual
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.Q_for_comb)

    def forward(self, bg_da, bg_q, h_da_last, h_q_last, h_da_0, h_q_0, MM, Mn, PEid, DEid):
        if MM is not None:
            tmp_q = torch.mm(MM, h_da_last)  # MM = nxm, h_da_last = mxd, tmp_q = nxd  和 h_q_last一样
        else:
            tmp_q = h_q_last  # tmp_q = nxd
        graph_level_q = self.pooling_q(bg_q, h_q_last)  # h_q_last = nxd, graph_level_q = b x d
        Q = self.Q_from_q(graph_level_q).reshape(-1, self.num_head,
                                                 2 * self.each_head)  # Q = b x nun_head x (2*each_head)

        # 小图q的计算：
        inx_q = inx_bg_num_nodes(bg_num_nodes=bg_q.batch_num_nodes())
        Q_for_q = Q[inx_q]  # Q_for_q = n x num_head x (2*each_head)
        Q_for_q = torch.split(Q_for_q, [self.each_head, self.each_head],
                              dim=2)  # Q_for_q = (n x num_head x each_head, n x num_head x each_head)
        feat_tmp_q = self.L(tmp_q).reshape(-1, self.num_head, self.each_head)  # feat_tmp_q = n x num_head x each_head
        el_src_q = (feat_tmp_q * Q_for_q[0]).sum(dim=-1).unsqueeze(-1)  # el_src_q = n x num_head x1
        er_dst_q = (feat_tmp_q * Q_for_q[1]).sum(dim=-1).unsqueeze(-1)  # er_dst_q = n x num_head x1
        final_h_q = self.att_comb_initAndT(Q=self.Q_for_comb, G=bg_q, M1=h_q_0,
                                           M2=h_q_last + (h_q_last - tmp_q))  # final_h_q = n x d
        final_h_q = self.L(final_h_q).reshape(-1, self.num_head, self.each_head)  # final_h_q = n x num_head x each_head
        bg_q.srcdata.update({'ft': final_h_q, 'el': el_src_q})
        bg_q.dstdata.update({'er': er_dst_q})
        bg_q.apply_edges(fn.u_add_v('el', 'er', 'e'))
        A_q = edge_softmax(bg_q, self.leaky_relu(bg_q.edata.pop('e')))  # A_q = q_e x num_head x 1
        bg_q.edata['a'] = self.drop_edge_q(A_q)
        bg_q.update_all(fn.u_mul_e('ft', 'a', 'm'),
                        fn.sum('m', 'ft'))
        h_q_next = bg_q.dstdata['ft']  # h_q_next = n x num_head x each_head
        h_q_next = h_q_next.reshape(-1, self.num_head * self.each_head)  # h_q_next = n x (num_head*each_head)
        h_q_next = self.end_FNN(h_q_next)  # nxd

        # 大图da的计算：
        inx_da = inx_bg_num_nodes(bg_num_nodes=bg_da.batch_num_nodes())
        Q_for_da = Q[inx_da]  # Q_for_da = m x num_head x (2*each_head)
        Q_for_da = torch.split(Q_for_da, [self.each_head, self.each_head],
                               dim=2)  # Q_for_da = (m x num_head x each_head, m x num_head x each_head)
        feat_da = self.L(h_da_last).reshape(-1, self.num_head, self.each_head)  # feat_da = m x num_head x each_head
        el_src_da = (feat_da * Q_for_da[0]).sum(dim=-1).unsqueeze(-1)  # el_src_da = m x num_head x1
        er_dst_da = (feat_da * Q_for_da[1]).sum(dim=-1).unsqueeze(-1)  # er_dst_da = m x num_head x1
        final_h_da = self.att_comb_initAndT(Q=self.Q_for_comb, G=bg_da, M1=h_da_0, M2=h_da_last)  # final_h_da = m x d
        final_h_da = self.L(final_h_da).reshape(-1, self.num_head,
                                                self.each_head)  # final_h_da = m x num_head x each_head
        bg_da.srcdata.update({'ft': final_h_da, 'el': el_src_da})
        bg_da.dstdata.update({'er': er_dst_da})
        bg_da.apply_edges(fn.u_add_v('el', 'er', 'e'))
        A_da = edge_softmax(bg_da, self.leaky_relu(bg_da.edata.pop('e')))  # A_da = da_e x num_head x 1
        bg_da.edata['a'] = self.drop_edge_da(A_da)
        bg_da.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        h_da_next = bg_da.dstdata['ft']  # h_da_next = m x num_head x each_head
        h_da_next = h_da_next.reshape(-1, self.num_head * self.each_head)  # h_da_next = m x (num_head*each_head)
        h_da_next = self.end_FNN(h_da_next)  # mxd

        if self.residual:
            h_da_next = h_da_last + h_da_next  # mxd
            h_q_next = h_q_last + h_q_next  # nxd

        PE = A_da[PEid]  # PE = num_PE x num_head x 1
        DE = A_da[DEid]  # DE = num_DE x num_head x 1
        PE = torch.sum(PE, dim=0)  # PE = num_head x 1
        DE = torch.sum(DE, dim=0)  # DE = num_head x 1
        PE_min_DE = (PE - DE) / len(Mn)  # PE_min_DE = num_head x 1  (a1-b0)
        mean_PE_min_DE = torch.mean(PE_min_DE, dim=0).reshape(1, 1)  # (a1-b0)  1x1
        return h_da_next, h_q_next, -1. * mean_PE_min_DE
