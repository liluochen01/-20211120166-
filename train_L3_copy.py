import torch.optim
from pylab import *
import os
import dataset_L2 as dataset
from sklearn import metrics
import torch.nn.functional as F
import network_L3 as network
from sklearn.metrics import precision_recall_curve
from torch import nn

import time
import math

start = time.time()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 计算预测的概率
        prob = torch.exp(-ce_loss)

        # 计算焦点损失
        focal_loss = self.alpha * (1 - prob) ** self.gamma * ce_loss

        # 根据'reduction'类型计算最终损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

"""
    修改train_L3.py训练模型的参数，分别设置：
        backbone = 'alexnet'
        backbone = 'resnet18'
        backbone = 'vgg16'
    训练三次模型，在./model/文件夹下得到
        model/alexnet/L3_alexnet_best_model.pkl
        model/resnet18/L3_resnet18_best_model.pkl
        model/vgg16/L3_vgg16_best_model.pkl
    最后，运行ensemble_L4.py
"""

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start = datetime.datetime.now()
print('Device:', device)

# 参数设置
is_sampling = 'no_sampler'  # 训练集采样模式： over_sampler-上采样  down_sampler-下采样  no_sampler-无采样
is_train = False  # True-训练模型  False-测试模型
save_model_name = 'model/vit/L3_vit_best_model (60).pkl'
is_pretrained = False  # 是否加载预训练权重
backbone = 'vit'  # 骨干网络：alexnet resnet18 vgg16 densenet inception
model_path = 'model/' + backbone  # 模型存储路径

# 训练参数设置
if backbone == 'vit':
    SIZE = 224
else:
    SIZE = 299
# SIZE = 299 if backbone == 'vit_base_patch16_224_in21k' else 224  # 图像进入网络的大小
BATCH_SIZE = 16  # batch_size数
NUM_CLASS = 2  # 分类数
EPOCHS = 25  # 迭代次数
train_batch = 10

# 进入工程路径并新建文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 进入工程路径
dataset.mkdir('model')  # 新建文件夹

if is_train:  # 训练模式
    PATH = 'data/labels.csv'
    TEST_PATH = ''
else:  # 测试模型
    PATH = 'data/labels.csv'
    TEST_PATH = 'data/exam_labels.csv'
    best_model_name = os.path.join(model_path, 'L3_%s_best_model.pkl' % backbone)
    print('best_model_name=', best_model_name)

# 加载数据
dataset.mkdir(model_path)
train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train=is_train,
                                                            is_sampling=is_sampling)
model = network.initialize_model(backbone, is_pretrained)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=6e-3)  # 更新所有层权重
criterion = FocalLoss(alpha=0.25, gamma=2)


# 训练模型
def train_vit(model):
    history_train = []
    history_valid = []
    history_auc = []
    best_auc = 0.

    # 动态lr设置
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(EPOCHS):
        correct = total = 0.
        loss_list = []
        # 为教学使用，仅选择部分数据进行训练，通过train_batch参数控制
        for batch_index, (batch_x, batch_y) in enumerate(train_loader, 0):
            if batch_index < train_batch:
                if torch.cuda.is_available():
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                model.train()
                # 优化过程
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                # 输出训练结果
                loss_list.append(loss.item())
                _, predicted = torch.max(output.data, 1)  # 返回每行的最大值
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        train_avg_acc = 100 * correct / total
        train_avg_loss = np.mean(loss_list)
        print('[Epoch=%d/%d]Train set: Avg_loss=%.4f, Avg_accuracy=%.4f%%' % (
            epoch + 1, EPOCHS, train_avg_loss, train_avg_acc))
        history_train.append((train_avg_loss, train_avg_acc))

        # change the learning rate by scheduler
        scheduler.step()

        # 验证集
        valid_pred, valid_true, auc, valid_acc, valid_avg_loss = valid_resnet(model)
        print('[Epoch=%d/%d]Validation set: Avg_loss=%.4f, Avg_accuracy=%.4f%%, AUC=%.4f' %
              (epoch + 1, EPOCHS, valid_avg_loss, valid_acc, auc))
        history_valid.append((valid_avg_loss, valid_acc))
        history_auc.append(auc)

        # 保存最优模型
        best_model_name = os.path.join(model_path, 'L3_%s_best_model.pkl' % backbone)
        if auc >= best_auc and epoch >= 10:
            print('>>>>>>>>>>>>>>Best model is %s' % (str(epoch + 1) + '.pkl'))
            torch.save(model.state_dict(), best_model_name)  # 训练多GPU，测试多GPU
            # torch.save(model.module.state_dict(), best_model_name)  # 训练多GPU，测试单GPU
            best_auc = auc

    print("Train finished!")
    print('Train running time = %s' % str(datetime.datetime.now() - start))
    print('Saving last model...')
    last_model_name = os.path.join(model_path, 'L3_%s_last_model.pkl' % backbone)
    torch.save(model.state_dict(), last_model_name)  # 训练多GPU，测试多GPU

    return best_model_name, history_train, history_valid, history_auc

def valid_resnet(model):
    # print('------ Validation Start -----')
    with torch.no_grad():
        model.eval()
        val_loss_list = []
        valid_pred = []
        valid_true = []
        valid_prob = np.empty(shape=[0, 2])  # 概率值

        for batch_index, (batch_valid_x, batch_valid_y) in enumerate(val_loader, 0):
            if torch.cuda.is_available():
                batch_valid_x, batch_valid_y = batch_valid_x.cuda(), batch_valid_y.cuda()
            output = model(batch_valid_x)
            _, batch_valid_pred = torch.max(output.data, 1)
            prob = F.softmax(output.data, dim=1)  # prob=softmax[[0.9,0.1],[0.8,0.2]]
            loss = criterion(output, batch_valid_y)
            val_loss_list.append(loss.item())
            valid_pred = np.hstack((valid_pred, batch_valid_pred.detach().cpu().numpy()))
            valid_true = np.hstack((valid_true, batch_valid_y.detach().cpu().numpy()))
            valid_prob = np.append(valid_prob, prob.detach().cpu().numpy(), axis=0)  # valid_prob=概率列表=[N*2]

        valid_avg_loss = np.mean(val_loss_list)
        valid_acc = 100 * metrics.accuracy_score(valid_true, valid_pred)
        valid_AUC = metrics.roc_auc_score(y_true=valid_true, y_score=valid_prob[:, 1])  # y_score=正例的概率=[N*1]
        # valid_AUC = metrics.roc_auc_score(y_true=valid_true, y_score=valid_pred)
        tn, fp, fn, tp = metrics.confusion_matrix(valid_true, valid_pred).ravel()
        valid_classification_report = metrics.classification_report(valid_true, valid_pred, digits=4)
    return valid_pred, valid_true, valid_AUC, valid_acc, valid_avg_loss

def adjust_threshold(model, test_loader, threshold=0.5):
    model.eval()
    probs = []
    true = []
    with torch.no_grad():
        for test_x, test_y in test_loader:
            if torch.cuda.is_available():
                images, labels = test_x.cuda(), test_y.cuda()
            else:
                images, labels = test_x, test_y
            output = model(images)
            prob = F.softmax(output, dim=1)[:, 1]
            probs.extend(prob.cpu().numpy())
            true.extend(labels.cpu().numpy())

    precision, recall, thresholds = precision_recall_curve(true, probs)
    # 找到最优阈值，比如F1最高点
    f1_scores = 2 * recall * precision / (recall + precision + 1e-6)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f'Best threshold: {best_threshold}')
    return best_threshold

def for_test_alexnet(model_name):
    print('------ Testing Start ------')
    print(model_name)
    model.load_state_dict(torch.load(model_name), False)
    test_pred = []
    test_true = []
    test_probs = []

    with torch.no_grad():
        model.eval()
        for test_x, test_y in test_loader:
            if torch.cuda.is_available():
                images, labels = test_x.cuda(), test_y.cuda()
            else:
                images, labels = test_x, test_y
            output = model(images)
            prob = F.softmax(output, dim=1)[:, 1]
            test_probs.extend(prob.cpu().numpy())
            _, predicted = torch.max(output.data, 1)
            test_pred = np.hstack((test_pred, predicted.detach().cpu().numpy()))
            test_true = np.hstack((test_true, labels.detach().cpu().numpy()))

    # 调整阈值
    best_threshold = adjust_threshold(model, test_loader)
    test_pred_adjusted = (np.array(test_probs) >= best_threshold).astype(int)

    images = test_loader.dataset.test_img
    test_acc = 100 * metrics.accuracy_score(test_true, test_pred_adjusted)
    test_classification_report = metrics.classification_report(test_true, test_pred_adjusted, digits=4)
    print('test_classification_report\n', test_classification_report)
    print('Accuracy of the network is: %.4f %%' % test_acc)
    return test_acc, images, test_true, test_pred_adjusted


if __name__ == '__main__':
    if is_train:
        best_model_name, history_train, history_valid, history_auc = train_vit(model)
        # 绘制训练集和验证集的loss、acc、AUC
        dataset.show_plot(history_train, history_valid, history_auc, model_path)
    else:
        for_test_alexnet(save_model_name)

    # print('头部ct运行时间level2：', time.time() - start)