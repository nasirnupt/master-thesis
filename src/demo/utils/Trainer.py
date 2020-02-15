import torch.nn as nn
import torch
import torch.functional as F
# coding: UTF-8
import numpy as np
import torch
from sklearn import metrics
import time
from tensorboardX import SummaryWriter
from datetime import timedelta
from tqdm import tqdm


def get_time_dif(start_time):
    """training time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class Trainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None, USE_CUDA=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.iterations = 0
        self.USE_CUDA = USE_CUDA

    def run(self, epochs=5):
        # 每一个epoch 就是一次train的过程
        # training epoch
        for i in range(1, epochs + 1):
            self.train()

    def train(self):
        # 从dataloader 中拿数据
        # take data from datalaoder
        for i, data in enumerate(self.dataset, self.iterations + 1):
            batch_input1, batch_input2, batch_target = data
            input1_var = batch_input1
            input2_var = batch_input2
            target_var = batch_target
            if self.USE_CUDA:
                input1_var = input1_var.cuda()
                input2_var = input2_var.cuda()
                target_var = target_var.cuda()

            # 每一次前馈就是一次函数闭包操作
            def closure():
                batch_output = self.model(input1_var,input2_var)
                loss = self.criterion(batch_output, target_var)
                print(loss)
                loss.backward()
                return loss

            # loss 返回,准备优化
            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.iterations += 1


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass



def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (train_1, labels) in enumerate(train_iter):
            outputs = model(train_1)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_pred(model, train_X, train_y, val_X, val_y, batch_size, epochs):
    valid_preds = np.zeros((val_X.size(0)))
    #test_preds = np.zeros(len(test_X))
    for e in range(epochs):
        start_time = time.time()

        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
        optimizer = torch.optim.Adam(model.parameters())

        train = torch.utils.data.TensorDataset(train_X,train_y)
        valid = torch.utils.data.TensorDataset(val_X,val_y)

        train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=64, shuffle=False)

        model.train()

        avg_loss = 0.
        for x_batch, y_batch in tqdm(train_loader, disable=True):
            y_pred = model(x_batch)
            #y_batch = y_batch.squeeze(-1)
            print(y_pred)
            y_pred = torch.max(y_pred,1)
            print(y_pred)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()
        avg_val_loss = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()
            y_pred = torch.max(y_pred,1)[1]
            print(y_pred)
            y_batch = y_batch.squeeze(-1)

            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            temp = y_pred.numpy()

            valid_preds[i * batch_size:(i + 1) * batch_size] = temp

        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
            e + 1, epochs, avg_loss, avg_val_loss, elapsed_time))

    return valid_preds

