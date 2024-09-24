import torch
import dataset_L2 as dataset
from transformers import ViTModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 参数设置
is_sampling = 'over_sampler'  # 训练集采样模式： over_sampler-上采样  down_sampler-下采样  no_sampler-无采样
is_train = True  # True-训练模型  False-测试模型
is_pretrained = True  # 是否加载预训练权重
backbone = 'vit_base_patch16_224_in21k'  # 骨干网络：alexnet resnet18 vgg16 densenet inception
model_path = 'model/' + backbone  # 模型存储路径

# 训练参数设置
if backbone == 'vit_base_patch16_224_in21k':
    SIZE = 224
else:
    SIZE = 299
# SIZE = 299 if backbone == 'vit_base_patch16_224_in21k' else 224  # 图像进入网络的大小
BATCH_SIZE = 16  # batch_size数
NUM_CLASS = 2  # 分类数
EPOCHS = 20  # 迭代次数
PATH = "data/labels.csv"
TEST_PATH = ''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset.mkdir(model_path)
train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train=is_train,
                                                            is_sampling=is_sampling)
vit_model = ViTModel.from_pretrained(r'E:\头部CT影像运动伪影识别v2.0\4.2项目源文件\Level3_4源文件\vit_base_patch16_224_in21k')
vit_model.to(device)
vit_model.eval()

# 用于存储提取的特征和标签
features = []
labels = []
train_batch = 10

# 提取特征
with torch.no_grad():
    for batch_index, (batch_x, batch_y) in enumerate(train_loader, 0):
        if batch_index < train_batch:
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                output = vit_model(batch_x)
                feature = output['pooler_output']
                features.append(feature)
                labels.append(batch_y)

# 将特征和标签转换为torch张量
features = torch.cat(features).cpu().numpy()
labels = torch.cat(labels).cpu().numpy()

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 初始化KNN分类器
knn = KNeighborsClassifier(n_neighbors=5)

# 训练KNN分类器
knn.fit(X_train, y_train)

# 在测试集上预测
y_pred = knn.predict(X_test)

# 评估KNN分类器的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN分类准确率: {accuracy * 100:.2f}%")