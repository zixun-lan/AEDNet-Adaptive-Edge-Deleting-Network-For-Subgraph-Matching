from dataSet import *
from layers import *
from model import CPADE_GAT


device = torch.device('cuda:0')
path_train = './SYNTHETIC/test/'
# path_test = './opendata/test/'
batch_size = 32
d_train = dgraph(root_dir=path_train)
data_loader_train = DataLoader(d_train, batch_size=batch_size, shuffle=False, collate_fn=collate)

md = CPADE_GAT(num_layers=6, h_dimension=32, categoricalTo0_or_continuousTo1=0, input_dimension=1, modeforsim_0forE_1forDot_2forleakyrelu=1,modefornorm_0forSoftmax_1forTau_2forMean=1, num_head=8, minmaxlist=[0, 7])
md = md.to(device)
md = md.train()

optimizer = torch.optim.Adam(md.parameters())
criterion = torch.nn.MSELoss().to(device)

epoch = 1
for i in range(epoch):
    epoch_loss = []
    F1_score = []
    for j, (bbg_da, bbg_q, b_mm_label, b_mask) in enumerate(data_loader_train):
        print(j)
        match_nodes, PEid, DEid = process_batch_matching_matrix(bbg_da, b_mm_label)
        bbg_da = bbg_da.to(device)
        bbg_q = bbg_q.to(device)
        b_mm_label = b_mm_label.to(device)
        b_mask = b_mask.to(device)












