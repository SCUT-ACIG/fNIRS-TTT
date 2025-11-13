import torch
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score
import numpy as np
from dataloader import Dataset, Load_Dataset_A, Load_Dataset_B, Load_Dataset_C
import os
import argparse
from model_init import init_model_MK2
parser = argparse.ArgumentParser(description="传入您的device_id, dataset_id(0->A, 1->B, 2->C), models_id(0->T, 1->PreT, 2->Ours)")
parser.add_argument('--device_id', type=str, default='4', help='传入您的device_id')
parser.add_argument('--models_id', type=int, default='0', help='传入您的models_id')
args = parser.parse_args()

def average_log(save_path):
    # 存储每个epoch的总和，用于求平均
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []

    sub_folders = [f.path for f in os.scandir(save_path) if f.is_dir()]  # 获取所有子文件夹
    num_subs = len(sub_folders)  # 记录子文件夹数量

    # 遍历所有子文件夹，读取train_loss_history.txt, train_acc_history.txt等文件
    for sub_folder in sub_folders:
        with open(os.path.join(sub_folder, 'train_loss_history.txt'), 'r') as f:
            train_loss_history = eval(f.read())  # 假设文件内容是一个列表
        
        with open(os.path.join(sub_folder, 'train_acc_history.txt'), 'r') as f:
            train_acc_history = eval(f.read())
        
        with open(os.path.join(sub_folder, 'test_loss_history.txt'), 'r') as f:
            test_loss_history = eval(f.read())
        
        with open(os.path.join(sub_folder, 'test_acc_history.txt'), 'r') as f:
            test_acc_history = eval(f.read())

        # 将当前子文件夹的数据累加到对应的总和列表中
        if len(train_loss_all) == 0:  # 第一次初始化每个列表
            train_loss_all = np.array(train_loss_history)
            train_acc_all = np.array(train_acc_history)
            test_loss_all = np.array(test_loss_history)
            test_acc_all = np.array(test_acc_history)
        else:
            train_loss_all += np.array(train_loss_history)
            train_acc_all += np.array(train_acc_history)
            test_loss_all += np.array(test_loss_history)
            test_acc_all += np.array(test_acc_history)

        # 计算平均值，并保留三位小数
        train_loss_avg = [round(loss, 3) for loss in (train_loss_all / num_subs).tolist()]
        train_acc_avg = [round(acc, 3) for acc in (train_acc_all / num_subs).tolist()]
        test_loss_avg = [round(loss, 3) for loss in (test_loss_all / num_subs).tolist()]
        test_acc_avg = [round(acc, 3) for acc in (test_acc_all / num_subs).tolist()]

    # 将最终结果保存到文件
    with open(os.path.join(save_path, 'final_train_loss_history.txt'), 'w') as f:
        f.write(str(train_loss_avg))
    
    with open(os.path.join(save_path, 'final_train_acc_history.txt'), 'w') as f:
        f.write(str(train_acc_avg))
    
    with open(os.path.join(save_path, 'final_test_loss_history.txt'), 'w') as f:
        f.write(str(test_loss_avg))
    
    with open(os.path.join(save_path, 'final_test_acc_history.txt'), 'w') as f:
        f.write(str(test_acc_avg))


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


def kfold_train_itself(device_id, models_id):
    

    # Training epochs
    EPOCH = 120

    # Device setting

    device_id = args.device_id

    # bs
    batch_size = 128

    # Select dataset
    dataset = ['A','B','C']
    models = ['fNIRS-T', 'fNIRS-PreT',  'CT-Net', 'fNIRS_TTT_M', 'fNIRS_TTT_L']
    models_id = args.models_id
    print(models[models_id])

    for index, ds in enumerate(dataset):
        dataset_id = index

        print(dataset[dataset_id])

        # Select the specified path
        # data_path = 'data'

        # Save file and avoid training file overwriting.
        save_path = 'save-addRes/' + dataset[dataset_id] + '/KFold/' + models[models_id]
        assert os.path.exists(save_path) is False, 'path is exist'
        os.makedirs(save_path)

        if dataset[dataset_id] == 'A':
            flooding_level = [0, 0, 0]
            if models[models_id] == 'fNIRS-T' or 1:
                feature, label = Load_Dataset_A("data/A", model='fNIRS-T')
            elif models[models_id] == 'fNIRS-PreT':
                feature, label = Load_Dataset_A("data/A", model='fNIRS-PreT')
        elif dataset[dataset_id] == 'B':
            if models[models_id] == 'fNIRS-T' or 1:
                flooding_level = [0.45, 0.40, 0.35]
            else:
                flooding_level = [0.40, 0.38, 0.35]
            feature, label = Load_Dataset_B("data/B")
        elif dataset[dataset_id] == 'C':
            flooding_level = [0.45, 0.40, 0.35]
            feature, label = Load_Dataset_C("data/C")
        
        _, _, channels, sampling_points = feature.shape

        feature = feature.reshape((label.shape[0], -1))
        # 5 × 5-fold-CV
        rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
        n_runs = 0

        result_acc = []
        result_pre = []
        result_rec = []
        result_f1  = []
        result_kap = []

        for train_index, test_index in rkf.split(feature):
            n_runs += 1
            print('======================================\n', n_runs)
            path = save_path + '/' + str(n_runs)
            assert os.path.exists(path) is False, 'sub-path is exist'
            os.makedirs(path)

            X_train = feature[train_index]
            y_train = label[train_index]
            X_test = feature[test_index]
            y_test = label[test_index]

            X_train = X_train.reshape((X_train.shape[0], 2, channels, -1))
            X_test = X_test.reshape((X_test.shape[0], 2, channels, -1))

            train_set = Dataset(X_train, y_train, transform=True)
            test_set = Dataset(X_test, y_test, transform=True)
            ########### fix seed ###########
            seed = 42
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            ################################
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True)

            # sample = train_set[0]
            # in_shape = sample.shape
            # -------------------------------------------------------------------------------------------------------------------- #
            device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
            

            net = init_model_MK2(dataset=dataset[dataset_id], model=models[models_id], device=device).to(device)

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

                    loss_steps.append(loss.item())
                    total += labels.shape[0]
                    pred = outputs.argmax(dim=1, keepdim=True)
                    train_running_acc += pred.eq(labels.view_as(pred)).sum().item()

                train_running_loss = float(np.mean(loss_steps))
                train_running_acc = 100 * train_running_acc / total
                # 将训练损失和准确率保存到对应列表
                train_loss_history.append(train_running_loss)
                train_acc_history.append(train_running_acc)
                print('[%d, %d] Train loss: %0.4f' % (n_runs, epoch, train_running_loss))
                print('[%d, %d] Train acc: %0.3f%%' % (n_runs, epoch, train_running_acc))

                # -------------------------------------------------------------------------------------------------------------------- #
                net.eval()
                test_running_acc = 0
                total = 0
                loss_steps = []
                y_label = y_pred = None
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
                    print('     [%d, %d] Test loss: %0.4f' % (n_runs, epoch, test_running_loss))
                    print('     [%d, %d] Test acc: %0.3f%%' % (n_runs, epoch, test_running_acc))

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
        average_log(save_path)


if __name__ == '__main__':
    kfold_train_itself(args.device_id, args.models_id)