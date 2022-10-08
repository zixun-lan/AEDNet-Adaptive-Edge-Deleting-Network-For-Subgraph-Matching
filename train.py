import time
from layers import *
from model import CPADE_GAT
from dataSet import dgraph, collate
from torch.utils.data import Dataset, DataLoader

num_layers = 5
h_dimension = 128
categoricalTo0_or_continuousTo1 = 0
input_dimension = 1
modeforsim_0forE_1forDot_2forleakyrelu = 1
modefornorm_0forSoftmax_1forTau_2forMean = 1
num_head = 8
minmaxlist = [0, 34]
residual = True
p_drop = 0.0
batchNorm = False
normAffine = False
negative_slope_in_norm = 0.2
tauForE_in_sim = 1
leaky_relu_GAT_rate = 2
drop_edge_p = 0.0
device = torch.device('cuda:0')
criterion = torch.nn.MSELoss().to(device)
batch_size = 500
eva_batch_size = 500
epoch = 5000
txt_path = './save/00record.txt'

with open(txt_path, 'w') as f:
    f.write('{} = {}\n'.format('num_layers', num_layers))
    f.write('{} = {}\n'.format('h_dimension', h_dimension))
    f.write('{} = {}\n'.format('num_head', num_head))
    f.write('{} = {}\n'.format('batchNorm', batchNorm))
    f.write('{} = {}\n'.format('leaky_relu_GAT_rate', leaky_relu_GAT_rate))
    f.write('{} = {}\n'.format('batch_size', batch_size))
    f.write('{} = {}\n'.format('eva_batch_size', eva_batch_size))
    f.write('\n')
    f.write('\n')

md = CPADE_GAT(num_layers=num_layers, h_dimension=h_dimension,
               categoricalTo0_or_continuousTo1=categoricalTo0_or_continuousTo1, input_dimension=input_dimension,
               modeforsim_0forE_1forDot_2forleakyrelu=modeforsim_0forE_1forDot_2forleakyrelu,
               modefornorm_0forSoftmax_1forTau_2forMean=modefornorm_0forSoftmax_1forTau_2forMean, num_head=num_head,
               minmaxlist=minmaxlist, residual=residual, p_drop=p_drop, batchNorm=batchNorm, normAffine=normAffine,
               negative_slope_in_norm=negative_slope_in_norm, tauForE_in_sim=tauForE_in_sim,
               leaky_relu_GAT_rate=leaky_relu_GAT_rate, drop_edge_p=drop_edge_p)
md = md.to(device)
optimizer = torch.optim.Adam(md.parameters())


path_train = './data/COX2/train/'
path_evaluation = './data/COX2/evaluation/'
path_test = './data/COX2/test/'

d_train = dgraph(root_dir=path_train)
data_loader_train = DataLoader(d_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
d_evaluation = dgraph(root_dir=path_evaluation)
data_loader_evaluation = DataLoader(d_evaluation, batch_size=eva_batch_size, shuffle=False, collate_fn=collate)
d_test = dgraph(root_dir=path_test)
data_loader_test = DataLoader(d_train, batch_size=eva_batch_size, shuffle=False, collate_fn=collate)

max_eva_F1 = 0
max_test_F1 = 0
min_eva_loss =100

for i in range(epoch):
    print('EPOCH: ', i)
    with open(txt_path, 'a') as f:
        f.write('EPOCH: {}\n'.format(i))
    t1 = time.time()
    epoch_loss_train = []
    F1_score_train = []
    md.train()
    for j, (bbg_da, bbg_q, b_mm_label, b_mask) in enumerate(data_loader_train):
        match_nodes, PEid, DEid = process_batch_matching_matrix(bbg_da, b_mm_label)
        bbg_da = bbg_da.to(device)
        bbg_q = bbg_q.to(device)
        b_mm_label = b_mm_label.to(device)
        b_mask = b_mask.to(device)

        MM, h_da_next, h_q_next, List_negative_DE, List_MM = md(bg_da=bbg_da, bg_q=bbg_q, h_da=bbg_da.ndata['x'],
                                                                h_q=bbg_q.ndata['x'], b_mask=b_mask, Mn=match_nodes,
                                                                PEid=PEid, DEid=DEid)
        loss = total_loss(lambda_step_T=0.8, lambda_step_T_DE=0.5, lambda_DE=0.5, mm_step_T=MM,
                          List_negative_DE=List_negative_DE, List_MM=List_MM, b_mm_label=b_mm_label,
                          criterion=criterion,
                          device=device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        f1_train = metric_f1(preticted_MM=MM, b_mm_label=b_mm_label)
        epoch_loss_train.append(float(loss.cpu().detach()))
        F1_score_train.append(f1_train)
        if j % 10 == 0:
            print('EPOCH: ', i, '| ', 'batch: ', j, '| ', 'batch_loss: ', float(loss.cpu().detach()), '| ', 'batch_F1: ', f1_train)
            with open(txt_path, 'a') as f:
                f.write('EPOCH: {} | batch: {} | batch_loss: {} | batch_F1: {}\n'.format(i, j, float(loss.cpu().detach()), f1_train))
    torch.save(md.state_dict(), './save/train00.pkl')
    t2 = time.time()
    print('*EPOCH: ', i, 'epoch_loss: ', np.mean(epoch_loss_train), '| ', 'epoch_F1: ', np.mean(F1_score_train))
    print('time: ', (t2 - t1) / 60)
    with open(txt_path, 'a') as f:
        f.write('*EPOCH: {} | epoch_loss: {} | epoch_F1: {}\n'.format(i, np.mean(epoch_loss_train), np.mean(F1_score_train)))
        f.write('time: {}\n'.format((t2 - t1) / 60))
    # 评估阶段
    epoch_loss_evaluation = []
    F1_score_evaluation = []
    md.eval()
    for j, (bbg_da, bbg_q, b_mm_label, b_mask) in enumerate(data_loader_evaluation):
        match_nodes, PEid, DEid = process_batch_matching_matrix(bbg_da, b_mm_label)
        bbg_da = bbg_da.to(device)
        bbg_q = bbg_q.to(device)
        b_mm_label = b_mm_label.to(device)
        b_mask = b_mask.to(device)

        with torch.no_grad():
            MM, h_da_next, h_q_next, List_negative_DE, List_MM = md(bg_da=bbg_da, bg_q=bbg_q, h_da=bbg_da.ndata['x'],
                                                                    h_q=bbg_q.ndata['x'], b_mask=b_mask, Mn=match_nodes,
                                                                    PEid=PEid, DEid=DEid)
            pass
        loss = total_loss(lambda_step_T=0.8, lambda_step_T_DE=0.5, lambda_DE=0.5, mm_step_T=MM,
                          List_negative_DE=List_negative_DE, List_MM=List_MM, b_mm_label=b_mm_label,
                          criterion=criterion,
                          device=device)

        f1_evaluation = metric_f1(preticted_MM=MM, b_mm_label=b_mm_label)
        epoch_loss_evaluation.append(float(loss.cpu().detach()))
        F1_score_evaluation.append(f1_evaluation)

    if max_eva_F1 < np.mean(F1_score_evaluation):
        max_eva_F1 = np.mean(F1_score_evaluation)
        torch.save(md.state_dict(), './save/eva_max_F1_00.pkl')
    if min_eva_loss > np.mean(epoch_loss_evaluation):
        min_eva_loss = np.mean(epoch_loss_evaluation)
        torch.save(md.state_dict(), './save/eva_min_loss_00.pkl')

    # 测试阶段
    F1_score_test = []
    md.eval()
    for j, (bbg_da, bbg_q, b_mm_label, b_mask) in enumerate(data_loader_test):
        match_nodes, PEid, DEid = process_batch_matching_matrix(bbg_da, b_mm_label)
        bbg_da = bbg_da.to(device)
        bbg_q = bbg_q.to(device)
        b_mm_label = b_mm_label.to(device)
        b_mask = b_mask.to(device)

        with torch.no_grad():
            MM, h_da_next, h_q_next, List_negative_DE, List_MM = md(bg_da=bbg_da, bg_q=bbg_q, h_da=bbg_da.ndata['x'],
                                                                    h_q=bbg_q.ndata['x'], b_mask=b_mask, Mn=match_nodes,
                                                                    PEid=PEid, DEid=DEid)
            pass

        f1_test = metric_f1(preticted_MM=MM, b_mm_label=b_mm_label)
        F1_score_test.append(f1_test)
    if max_test_F1 < np.mean(F1_score_test):
        max_test_F1 = np.mean(F1_score_test)
        torch.save(md.state_dict(), './save/test_max_F1_00.pkl')
    t3 = time.time()
    print('*!* COMPLETED EPOCH: ', i, '| ', 'eva_loss: ', np.mean(epoch_loss_evaluation), '| ', 'current_eva_F1: ', np.mean(F1_score_evaluation), 'max_eva_F1: ', max_eva_F1, '| ', 'max_test_F1: ', max_test_F1)
    print('time: ', (t3 - t1) / 60)
    print(md.matchingmatrix.normalization_layer.w)
    with open(txt_path, 'a') as f:
        f.write('*!* COMPLETED EPOCH: {} | eva_loss: {} | current_eva_F1: {} | max_eva_F1: {} | max_test_F1: {}\n'.format(i, np.mean(epoch_loss_evaluation), np.mean(F1_score_evaluation), max_eva_F1, max_test_F1))
        f.write('time: {}\n'.format((t3 - t1) / 60))
        f.write('w: {}\n'.format(md.matchingmatrix.normalization_layer.w.cpu().detach().numpy()[0]))
        f.write('\n')








