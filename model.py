from layers import CPADE_GAT_layer, MatchingMatrix, project_layers
import torch


class CPADE_GAT(torch.nn.Module):
    def __init__(self, num_layers, h_dimension, categoricalTo0_or_continuousTo1, input_dimension, modeforsim_0forE_1forDot_2forleakyrelu,modefornorm_0forSoftmax_1forTau_2forMean, num_head, minmaxlist=None, residual=True, p_drop=0, batchNorm=True, normAffine=False, negative_slope_in_norm=0.2, tauForE_in_sim=1, leaky_relu_GAT_rate=2, drop_edge_p=0):
        super(CPADE_GAT, self).__init__()
        self.projection = project_layers(categoricalTo0_or_continuousTo1=categoricalTo0_or_continuousTo1, input_dimension=input_dimension, h_dimension=h_dimension, p_drop=p_drop, minmaxlist=minmaxlist,
                 batchNorm=batchNorm, normAffine=normAffine)
        self.matchingmatrix = MatchingMatrix(modeforsim_0forE_1forDot_2forleakyrelu=modeforsim_0forE_1forDot_2forleakyrelu, modefornorm_0forSoftmax_1forTau_2forMean=modefornorm_0forSoftmax_1forTau_2forMean, h_dimension=h_dimension,
                 negative_slope=negative_slope_in_norm, tauForE=tauForE_in_sim)
        self.layers = torch.nn.ModuleList([CPADE_GAT_layer(h_dimension=h_dimension, num_head=num_head, residual=residual, leaky_relu_GAT_rate=leaky_relu_GAT_rate, batchNorm=batchNorm, normAffine=normAffine, p_drop=p_drop, drop_edge_p=drop_edge_p) for i in range(num_layers)])

    def forward(self, bg_da, bg_q, h_da, h_q, b_mask, Mn, PEid, DEid):
        h_da_0 = h_da_next = self.projection(h_da)
        h_q_0 = h_q_next = self.projection(h_q)
        MM = None
        List_MM = []
        List_negative_DE = []  # -(1-0)
        for i in self.layers:
            h_da_next, h_q_next, negative_DE = i(bg_da=bg_da, bg_q=bg_q, h_da_last=h_da_next, h_q_last=h_q_next, h_da_0=h_da_0, h_q_0=h_q_0, MM=MM, Mn=Mn, PEid=PEid, DEid=DEid)
            MM = self.matchingmatrix(h_da=h_da_next, h_q=h_q_next, mask=b_mask)
            List_negative_DE.append(negative_DE)
            List_MM.append(MM)
        return MM, h_da_next, h_q_next, List_negative_DE, List_MM











