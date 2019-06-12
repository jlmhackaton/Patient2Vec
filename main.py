# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:49:49 2018

@author: Zhiyong
"""

from Patient2Vec import *

TIME_COL_HDR = 'Time_inED'
SUBJECT_COL_HDR = 'SUBJECT_ID'
SUBJECT_COLTAG_HDR = 'HADM_ID'
TAG_HDR = 'INTERPRETATION'

def PrepareDataset(features_data_path, \
                   tags_data_path,\
                   BATCH_SIZE = 40, \
                   seq_len = 10, \
                   pred_len = 1, \
                   train_propotion = 0.7, \
                   valid_propotion = 0.2,
                   shuffle=False):
    """ Prepare training and testing datasets and dataloaders.
    
    Convert admissions table to training and testing dataset.
    The vertical axis of admissions_pd is the admission time axis and the horizontal axis
    is the features axis.
    
    Args:
        admissions_pd: a Matrix containing spatial-temporal speed data for a network
        seq_len: length of input sequence
        pred_len: length of predicted sequence
    Returns:
        Training dataloader
        Testing dataloader
    """

    admissions_pd = pd.read_csv(features_data_path) #header = None, names=["Eloss", "entropy", "loss", "tErr", "rotErr", "r1", "r2", "r3", "tx", "ty", "tz" ], index_col=False, skiprows=0, delimiter=" "
    admissions_byID = admissions_pd.groupby(SUBJECT_COL_HDR)

    tag_pd = pd.read_csv(tags_data_path)
    tags_byID = tag_pd.groupby(SUBJECT_COLTAG_HDR)

    max_admin_per_patient = 0
    features_list = []
    tag_list = []
    for group, tag_it in zip(admissions_byID, tags_byID):
        date_sorted = group[1].sort_values(by=[TIME_COL_HDR])
        features = date_sorted[admissions_pd.columns[2:]].values
        features_list.append(features)
        max_admin_per_patient = max(max_admin_per_patient, features.shape[0])

        tag = tag_it[1][[TAG_HDR]].values[0]
        tag_list.append(tag)

    print('Maximum number of admissions per patient is ' + str(max_admin_per_patient))
    sample_size = len(features_list)
    features_nd = np.zeros([sample_size,max_admin_per_patient, features_list[0].shape[-1]])

    for idx, patient_adm in enumerate(features_list):
        h = patient_adm.shape[0]
        features_nd[idx, 0:h] = patient_adm

    # shuffle = True
    if shuffle: # doesn't work !! need to debug
        # shuffle and split the dataset to training and testing datasets
        print('Start to shuffle and split dataset ...')
        index = np.arange(sample_size, dtype = int)
        np.random.seed(1024)
        np.random.shuffle(index)

        features_nd = features_nd[index]
        tag_list = np.array(tag_list)
        tag_list = tag_list[index]

    if False:
        X_last_obsv = X_last_obsv[index]
        Mask = Mask[index]
        Delta = Delta[index]
        features_list = np.expand_dims(features_list, axis=1)
        X_last_obsv = np.expand_dims(X_last_obsv, axis=1)
        Mask = np.expand_dims(Mask, axis=1)
        Delta = np.expand_dims(Delta, axis=1)
        dataset_agger = np.concatenate((features_list, X_last_obsv, Mask, Delta), axis = 1)

    train_index = int(np.floor(sample_size * train_propotion))
    valid_index = int(np.floor(sample_size * (train_propotion + valid_propotion)))

    train_data, train_label = features_nd[:train_index], tag_list[:train_index]
    valid_data, valid_label = features_nd[train_index:valid_index], tag_list[train_index:valid_index]
    test_data, test_label = features_nd[valid_index:], tag_list[valid_index:]
    
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)
    
    train_dataset = utils.TensorDataset(train_data, train_label)
    valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)
    
    train_dataloader = utils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # X_mean = np.mean(features_list, axis = 0)
    
    print('Finished preprocessing')
    
    return train_dataloader, valid_dataloader, test_dataloader


def Train_Model(model, train_dataloader, valid_dataloader, batch_size, num_epochs = 300, patience=10, min_delta = 0.00001):
    
    print('Model Structure: ', model)
    print('Start Training ... ')
    output_last = True
    
    #model.cuda()
    #
    # if (type(model) == nn.modules.container.Sequential):
    #     output_last = model[-1].output_last
    #     print('Output type dermined by the last layer')
    # else:
    #     output_last = model.output_last
    #     print('Output type dermined by the model')
        
    # loss_MSE = torch.nn.MSELoss()
    # loss_L1 = torch.nn.L1Loss()
    criterion = nn.BCELoss()
    
    learning_rate = 0.0001
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate, alpha=0.99)
    use_gpu = torch.cuda.is_available()
    
    # interval = 100
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []
    
    cur_time = time.time()
    pre_time = time.time()
    
    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    for epoch in range(num_epochs):
        
        trained_number = 0
        
        valid_dataloader_iter = iter(valid_dataloader)
        
        losses_epoch_train = []
        losses_epoch_valid = []
        
        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else: 
                inputs, labels = Variable(inputs), Variable(labels)
            
            model.zero_grad()

            # TODO: extract them
            inputs_other = []#(age, gender, previous_hospitalization_history)
            outputs, alpha, beta = model(inputs, inputs_other, batch_size)

            # if output_last:
            #     # loss_train = loss_MSE(torch.squeeze(outputs), torch.squeeze(labels))
            loss_train = get_loss(outputs, labels, criterion=criterion, mtr=beta)
            # else:
            #     full_labels = torch.cat((inputs[:,1:,:], labels), dim = 1)
            #     loss_train = loss_MSE(outputs, full_labels)
        
            losses_train.append(loss_train.data)
            losses_epoch_train.append(loss_train.data)
            
            optimizer.zero_grad()
            
            loss_train.backward()
            
            optimizer.step()
            
             # validation 
            try: 
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)
            
            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else: 
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
                
            model.zero_grad()
            # TODO: extract them
            inputs_other = []  # (age, gender, previous_hospitalization_history)
            inputs_other_val = []  # (age, gender, previous_hospitalization_history)
            outputs_val, alpha_val, beta_val = model(inputs_val, inputs_other_val, batch_size)
            
            # if output_last:
            #     # loss_valid = loss_MSE(torch.squeeze(outputs_val), torch.squeeze(labels_val))
            loss_valid = get_loss(outputs_val, labels_val, criterion=criterion, mtr=beta_val)
            # else:
            #     full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val), dim = 1)
            #     loss_valid = loss_MSE(outputs_val, full_labels_val)

            losses_valid.append(loss_valid.data)
            losses_epoch_valid.append(loss_valid.data)
            
            # output
            trained_number += 1
            
        avg_losses_epoch_train = sum(losses_epoch_train).cpu().numpy() / float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid).cpu().numpy() / float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)
        
        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                torch.save(model.state_dict(), 'best_model.pt')
                min_loss_epoch_valid = avg_losses_epoch_valid 
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break
        
        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, time: {}, best model: {}'.format( \
                    epoch, \
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    np.around([cur_time - pre_time] , decimals=2),\
                    is_best_model) )
        pre_time = cur_time
                
    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid]


def Test_Model(model, test_dataloader, batch_size):
    
    # if (type(model) == nn.modules.container.Sequential):
    #     output_last = model[-1].output_last
    # else:
    #     output_last = model.output_last
    
    inputs, labels = next(iter(test_dataloader))

    cur_time = time.time()
    pre_time = time.time()
    
    use_gpu = torch.cuda.is_available()
    
    # loss_MSE = torch.nn.MSELoss()
    # loss_L1 = torch.nn.MSELoss()
    criterion = nn.BCELoss()

    tested_batch = 0
    
    # losses_mse = []
    losses_bce = []
    # losses_l1 = []
    MAEs = []
    MAPEs = []

    
    for data in test_dataloader:
        inputs, labels = data
        
        if inputs.shape[0] != batch_size:
            continue
    
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else: 
            inputs, labels = Variable(inputs), Variable(labels)

        # TODO: extract them
        inputs_other = []  # (age, gender, previous_hospitalization_history)
        outputs, alpha, beta = model(inputs, inputs_other, batch_size)
    
        # loss_MSE = torch.nn.MSELoss()
        # loss_L1 = torch.nn.L1Loss()
        
        loss_bce = get_loss(outputs, labels, criterion, beta)
        # if output_last:
        #     loss_mse = loss_MSE(torch.squeeze(outputs), torch.squeeze(labels))
        #     loss_l1 = loss_L1(torch.squeeze(outputs), torch.squeeze(labels))
        MAE = torch.mean(torch.abs(torch.squeeze(outputs) - torch.squeeze(labels)))
        MAPE = torch.mean(torch.abs(torch.squeeze(outputs) - torch.squeeze(labels)) / torch.squeeze(labels))
        # else:
        #     loss_mse = loss_MSE(outputs[:,-1,:], labels)
        #     loss_l1 = loss_L1(outputs[:,-1,:], labels)
        #     MAE = torch.mean(torch.abs(outputs[:,-1,:] - torch.squeeze(labels)))
        #     MAPE = torch.mean(torch.abs(outputs[:,-1,:] - torch.squeeze(labels)) / torch.squeeze(labels))
            
        # losses_mse.append(loss_mse.data)
        # losses_l1.append(loss_l1.data)
        losses_bce.append(loss_bce.data)
        MAEs.append(MAE.data)
        MAPEs.append(MAPE.data)
        
        tested_batch += 1
    
        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_bce: {}, time: {}'.format( \
                  tested_batch * batch_size, \
                  np.around([loss_bce.data[0]], decimals=8), \
                  # np.around([loss_mse.data[0]], decimals=8), \
                  np.around([cur_time - pre_time], decimals=8) ) )
            pre_time = cur_time
    losses_bce = np.array(losses_bce)
    # losses_l1 = np.array(losses_l1)
    # losses_mse = np.array(losses_mse)
    MAEs = np.array(MAEs)
    MAPEs = np.array(MAPEs)
    
    mean_l1 = np.mean(losses_bce)
    std_l1 = np.std(losses_bce)
    MAE_ = np.mean(MAEs)
    MAPE_ = np.mean(MAPEs) * 100
    
    print('Tested: bce_mean: {}, bce_std: {}, MAE: {} MAPE: {}'.format(mean_l1, std_l1, MAE_, MAPE_))
    return [losses_bce, mean_l1, std_l1]


if __name__ == "__main__":

    # TODO: pick demographic features and remove them from feature. remove the comment that ignores them
    # TODO: add the dimension of the added demographic features to the last linear layer
    # TODO: remove the repeat trick
    # TODO: adapt the test/train code
    # TODO: save & load the model and data
    ##########################################################3
    ###########         configurations
    ###########################################################
    features_datapath = './data/input.csv'
    tags_datapath = './data/output.csv'
    load_model = False
    batch_size = 1
    shuffle = False     # shuffle dataset

    ############################################################

    train_dataloader, valid_dataloader, test_dataloader = PrepareDataset(features_data_path=features_datapath,
                                tags_data_path=tags_datapath, BATCH_SIZE=batch_size, shuffle =shuffle)
    
    inputs, labels = next(iter(train_dataloader))
    [batch_size, seq_len, input_dim] = inputs.size()
    output_dim = labels.shape[-1]

    pat2vec = Patient2Vec(input_size=input_dim, hidden_size=256, n_layers=1, att_dim=1, initrange=1,
                          output_size=output_dim, rnn_type='GRU', seq_len=seq_len, pad_size=2,
                          n_filters=3, bi=True)
    if not load_model:
        best_grud, losses_grud = Train_Model(pat2vec, train_dataloader, valid_dataloader, num_epochs = 40, batch_size=batch_size)
        [losses_bce, mean_l1, std_l1] = Test_Model(best_grud, test_dataloader, batch_size=batch_size)
    else:
        pat2vec.load_state_dict(torch.load('best_model.pt'))
        # pat2vec.eval()
        [losses_mse, mean_l1, std_l1] = Test_Model(pat2vec, test_dataloader, batch_size=batch_size)