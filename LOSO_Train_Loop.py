import torch
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score
from dataloader import Dataset, Load_Dataset_A, Load_Dataset_B, Load_Dataset_C
from LOSO_Split import Split_Dataset_A, Split_Dataset_B, Split_Dataset_C
import os
import argparse
from model_init import init_model_MK2


parser = argparse.ArgumentParser(description="传入您的device_id, dataset_id(0->A, 1->B, 2->C), models_id(0->T, 1->PreT, 2->Ours)")
parser.add_argument('--device_id', type=str, default='1', help='传入您的device_id')
# parser.add_argument('--dataset_id', type=int, default='0', help='传入您的dataset_id(0->A, 1->B, 2->C)')
parser.add_argument('--models_id', type=int, default='14', help='传入您的models_id(0->T, 1->PreT, 2->Ours)')
args = parser.parse_args()


class LabelSmoothing(torch.nn.Module):
    """NLL loss with label smoothing."""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def LOSO_Train_itself(device_id, models_id):
    # Training epochs
    EPOCH = 120

    # Device setting
    device_id = args.device_id

    # Select dataset by setting dataset_id

    dataset = ['A','B','C']

    # dataset_id = args.dataset_id

    # bs
    batch_size = 128

    models = ['fNIRS-T', 'fNIRS-PreT', 'CT-Net', 'fNIRS_TTT_M', 'fNIRS_TTT_L']
    models_id = args.models_id
    print(models[models_id])

    # Select the specified path
    # data_path = 'data'
    for index, ds in enumerate(dataset):
        dataset_id = index
        print(dataset[dataset_id])
        # Save file and avoid training file overwriting.
        save_path = 'save/' + dataset[dataset_id] + '/LOSO/' + models[models_id]
        assert os.path.exists(save_path) is False, 'path is exist'
        os.makedirs(save_path)

        # Load dataset, set flooding levels and number of Subjects. Different models may have different flooding levels.
        # if dataset[dataset_id] == 'A':
        #     flooding_level = [0, 0, 0]
        #     Subjects = 8
        #     if models[models_id] == 'fNIRS-T' or models[models_id].startswith('fNIRS_'):
        #         feature, label = Load_Dataset_A("data/A", model='fNIRS-T')
        #     elif models[models_id] == 'fNIRS-PreT':
        #         feature, label = Load_Dataset_A("data/A", model='fNIRS-PreT')
        # elif dataset[dataset_id] == 'B':
        #     if models[models_id] == 'fNIRS-T' or models[models_id].startswith('fNIRS_'):
        #         flooding_level = [0.45, 0.40, 0.35]
        #     else:
        #         flooding_level = [0.40, 0.38, 0.35]
        #     Subjects = 28
        #     feature, label = Load_Dataset_B("data/B")
        # elif dataset[dataset_id] == 'C':
        #     flooding_level = [0.45, 0.40, 0.35]
        #     Subjects = 30
        #     feature, label = Load_Dataset_C("data/C")

        if dataset[dataset_id] == 'A':
            flooding_level = [0, 0, 0]
            Subjects = 8
            if models[models_id] == 'fNIRS-T' or 1:
                feature, label = Load_Dataset_A("data/A", model='fNIRS-T')
            elif models[models_id] == 'fNIRS-PreT':
                feature, label = Load_Dataset_A("data/A", model='fNIRS-PreT')
        elif dataset[dataset_id] == 'B':
            Subjects = 28
            if models[models_id] == 'fNIRS-T' or 1:
                flooding_level = [0.45, 0.40, 0.35]
            else:
                flooding_level = [0.40, 0.38, 0.35]
            feature, label = Load_Dataset_B("data/B")
        elif dataset[dataset_id] == 'C':
            Subjects = 30
            flooding_level = [0.45, 0.40, 0.35]
            feature, label = Load_Dataset_C("data/C")

        _, _, channels, sampling_points = feature.shape

        result_acc = []
        result_pre = []
        result_rec = []
        result_f1  = []
        result_kap = []

        for sub in range(1, Subjects+1):
            # Spilt dataset, one subject's data is test set, and the rest is training set.
            if dataset[dataset_id] == 'A':
                X_train, y_train, X_test, y_test = Split_Dataset_A(sub, feature, label, channels)
            elif dataset[dataset_id] == 'B':
                X_train, y_train, X_test, y_test = Split_Dataset_B(sub, feature, label, channels)
            elif dataset[dataset_id] == 'C':
                X_train, y_train, X_test, y_test = Split_Dataset_C(sub, feature, label, channels)

            path = save_path + '/' + str(sub)
            assert os.path.exists(path) is False, 'sub-path is exist'
            os.makedirs(path)

            train_set = Dataset(X_train, y_train, transform=True)
            test_set = Dataset(X_test, y_test, transform=True)
            ########### fix seed ###########
            seed = 42
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ################################
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

            # -------------------------------------------------------------------------------------------------------------------- #
            device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
            # if dataset[dataset_id] == 'A':
            #     if models[models_id] == 'fNIRS-T':
            #         net = fNIRS_T(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
            #     elif models[models_id] == 'fNIRS_TTT':
            #         net = fNIRS_TTT(n_class=2, sampling_point=sampling_points, 
            #                         dim=64, depth=6, heads=8, mlp_dim=64, device=device,
            #                         dataset=dataset[dataset_id], batch_size=batch_size).to(device)    
            #     elif models[models_id] == 'fNIRS_TTTM':
            #         net = fNIRS_TTTM(n_class=2, sampling_point=sampling_points, 
            #                         dim=64, depth=6, heads=8, mlp_dim=64, device=device,
            #                         dataset=dataset[dataset_id], batch_size=batch_size).to(device)
            #     elif models[models_id] == 'fNIRS_TTT_KTX2':
            #         net = fNIRS_TTT_KTX2(n_class=2, sampling_point=sampling_points, 
            #                         dim=64, depth=6, heads=8, mlp_dim=64, device=device,
            #                         dataset=dataset[dataset_id], batch_size=batch_size).to(device)                 

            #     elif models[models_id] == 'fNIRS-PreT':
            #         net = fNIRS_PreT(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device) 
            # elif dataset[dataset_id] == 'B':
            #     if models[models_id] == 'fNIRS-T':
            #         net = fNIRS_T(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
            #     elif models[models_id] == 'fNIRS_TTT':
            #         net = fNIRS_TTT(n_class=2, sampling_point=sampling_points, 
            #                         dim=64, depth=6, heads=8, mlp_dim=64, device=device,
            #                         dataset=dataset[dataset_id], batch_size=batch_size).to(device) 
            #     elif models[models_id] == 'fNIRS_TTTM':
            #         net = fNIRS_TTTM(n_class=2, sampling_point=sampling_points, 
            #                         dim=64, depth=6, heads=8, mlp_dim=64, device=device,
            #                         dataset=dataset[dataset_id], batch_size=batch_size).to(device) 
            #     elif models[models_id] == 'fNIRS_TTT_KTX2':
            #         net = fNIRS_TTT_KTX2(n_class=2, sampling_point=sampling_points, 
            #                         dim=64, depth=6, heads=8, mlp_dim=64, device=device,
            #                         dataset=dataset[dataset_id], batch_size=batch_size).to(device)  
            #     elif models[models_id] == 'fNIRS-PreT':
            #         net = fNIRS_PreT(n_class=2, sampling_point=sampling_points, dim=64, depth=6, heads=8, mlp_dim=64).to(device)
            # elif dataset[dataset_id] == 'C':
            #     if models[models_id] == 'fNIRS-T':
            #         net = fNIRS_T(n_class=3, sampling_point=sampling_points, dim=128, depth=6, heads=8, mlp_dim=64).to(device)
            #     elif models[models_id] == 'fNIRS_TTT':
            #         net = fNIRS_TTT(n_class=3, sampling_point=sampling_points, 
            #                         dim=64, depth=6, heads=8, mlp_dim=64, device=device,
            #                         dataset=dataset[dataset_id], batch_size=batch_size).to(device)
            #     elif models[models_id] == 'fNIRS_TTTM':
            #         net = fNIRS_TTTM(n_class=3, sampling_point=sampling_points, 
            #                         dim=64, depth=6, heads=8, mlp_dim=64, device=device,
            #                         dataset=dataset[dataset_id], batch_size=batch_size).to(device)
            #     elif models[models_id] == 'fNIRS-PreT':
            #         net = fNIRS_PreT(n_class=3, sampling_point=sampling_points, dim=128, depth=6, heads=8, mlp_dim=64).to(device)
            #     elif models[models_id] == 'fNIRS_TTT_KTX2':
            #         net = fNIRS_TTT_KTX2(n_class=3, sampling_point=sampling_points, 
            #                         dim=64, depth=6, heads=8, mlp_dim=64, device=device,
            #                         dataset=dataset[dataset_id], batch_size=batch_size).to(device)  
            
            # net = generate_model(models[models_id], dataset[dataset_id]).to(device)
            net = init_model_MK2(dataset=dataset[dataset_id], model=models[models_id], device=device)
            criterion = LabelSmoothing(0.1)
            optimizer = torch.optim.AdamW(net.parameters())
            lrStep = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
            # -------------------------------------------------------------------------------------------------------------------- #
            test_max_acc = 0
            train_loss_history = []
            train_acc_history = []
            test_loss_history = []
            test_acc_history = []
            for epoch in range(EPOCH):
                net.train()
                train_running_acc = 0
                total = 0
                loss_steps = []
                for i, data in enumerate(train_loader):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, labels.long())

                    # Piecewise decay flooding. b is flooding level, b = 0 means no flooding
                    if epoch < 30:
                        b = flooding_level[0]
                    elif epoch < 50:
                        b = flooding_level[1]
                    else:
                        b = flooding_level[2]

                    # flooding
                    loss = (loss - b).abs() + b
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()

                    loss_steps.append(loss.item())
                    total += labels.shape[0]
                    pred = outputs.argmax(dim=1, keepdim=True)
                    train_running_acc += pred.eq(labels.view_as(pred)).sum().item()

                train_running_loss = float(np.mean(loss_steps))
                train_running_acc = 100 * train_running_acc / total
                # 将训练损失和准确率保存到对应列表
                train_loss_history.append(train_running_loss)
                train_acc_history.append(train_running_acc)
                print('[%d, %d] Train loss: %0.4f' % (sub, epoch, train_running_loss))
                print('[%d, %d] Train acc: %0.3f%%' % (sub, epoch, train_running_acc))

                # -------------------------------------------------------------------------------------------------------------------- #
                net.eval()
                test_running_acc = 0
                total = 0
                loss_steps = []
                with torch.no_grad():
                    for data in test_loader:
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = net(inputs)
                        loss = criterion(outputs, labels.long())

                        loss_steps.append(loss.item())
                        total += labels.shape[0]
                        pred = outputs.argmax(dim=1, keepdim=True)
                        test_running_acc += pred.eq(labels.view_as(pred)).sum().item()

                    test_running_acc = 100 * test_running_acc / total
                    test_running_loss = float(np.mean(loss_steps))
                    test_loss_history.append(test_running_loss)
                    test_acc_history.append(test_running_acc)
                    print('     [%d, %d] Test loss: %0.4f' % (sub, epoch, test_running_loss))
                    print('     [%d, %d] Test acc: %0.3f%%' % (sub, epoch, test_running_acc))

                    if test_running_acc > test_max_acc:
                        test_max_acc = test_running_acc
                        y_label = labels.to('cpu')
                        y_pred = pred.to('cpu')
                        acc = accuracy_score(y_label, y_pred)
                        if dataset[dataset_id] == 'C':
                            # Multi-classification using macro mode
                            precision = precision_score(y_label, y_pred, average='macro')
                            recall = recall_score(y_label, y_pred, average='macro')
                            f1 = f1_score(y_label, y_pred, average='macro')
                        else:
                            precision = precision_score(y_label, y_pred)
                            recall = recall_score(y_label, y_pred)
                            f1 = f1_score(y_label, y_pred)
                        kappa_value = cohen_kappa_score(y_label, y_pred)
                        if 1: # 保存模型和数值
                            torch.save(net.state_dict(), path + '/model.pt')
                            test_save = open(path + '/test_acc.txt', "w")
                            test_save.write("%.3f" % (test_running_acc))
                            test_save.close()

                            test_save = open(path + '/test_acc2.txt', "w")
                            test_save.write("%.3f" % (acc))
                            test_save.close()

                            test_save = open(path + '/test_precision.txt', "w")
                            test_save.write("%.3f" % (precision))
                            test_save.close()

                            test_save = open(path + '/test_recall.txt', "w")
                            test_save.write("%.3f" % (recall))
                            test_save.close()

                            test_save = open(path + '/test_f1.txt', "w")
                            test_save.write("%.3f" % (f1))
                            test_save.close()

                            test_save = open(path + '/test_kappa.txt', "w")
                            test_save.write("%.3f" % (kappa_value))
                            test_save.close()

                lrStep.step()
            result_acc.append(acc)
            result_pre.append(precision)
            result_rec.append(recall)
            result_f1.append(f1)
            result_kap.append(kappa_value)

            test_save = open(path + '/train_loss_history.txt', 'w')
            test_save.write(str(train_loss_history))
            test_save.close()
            test_save = open(path + '/train_acc_history.txt', 'w')
            test_save.write(str(train_acc_history))
            test_save.close()
            test_save = open(path + '/test_loss_history.txt', 'w')
            test_save.write(str(test_loss_history))
            test_save.close()
            test_save = open(path + '/test_acc_history.txt', 'w')
            test_save.write(str(test_acc_history))
            test_save.close()

        result_acc = np.array(result_acc)
        acc_mean, acc_std = float(np.mean(result_acc)), float(np.std(result_acc))
        result_pre = np.array(result_pre)
        pre_mean, pre_std = float(np.mean(result_pre)), float(np.std(result_pre))
        result_rec = np.array(result_rec)
        rec_mean, rec_std = float(np.mean(result_rec)), float(np.std(result_rec))
        result_f1 = np.array(result_f1)
        f1_mean = float(np.mean(result_f1))
        result_kap = np.array(result_kap)
        kap_mean = float(np.mean(result_kap))
        print("\n"+dataset[dataset_id])
        print(models[models_id])
        print('acc_mean = %.2f, std = %.2f' % (acc_mean * 100, acc_std * 100))
        print('pre_mean = %.2f, std = %.2f' % (pre_mean * 100, pre_std * 100))
        print('rec_mean = %.2f, std = %.2f' % (rec_mean * 100, rec_std * 100))
        print('f1_mean = %.2f' % (f1_mean * 100))
        print('kap_mean = %.2f' % kap_mean)
        
        test_save = open(save_path + '/result.txt', "w")
        test_save.write('acc_mean = %.2f, std = %.2f \n' % (acc_mean * 100, acc_std * 100))
        test_save.write('pre_mean = %.2f, std = %.2f \n' % (pre_mean * 100, pre_std * 100))
        test_save.write('rec_mean = %.2f, std = %.2f \n' % (rec_mean * 100, rec_std * 100))
        test_save.write('f1_mean = %.2f \n' % (f1_mean * 100))
        test_save.write('kap_mean = %.2f \n' % kap_mean)
        test_save.close()

        # average_log(save_path)



if __name__ == '__main__':
    LOSO_Train_itself(args.device_id, args.models_id)