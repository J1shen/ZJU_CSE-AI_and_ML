###############################################################################
# 重要: 请务必把任务(jobs)中需要保存的文件存放在 results 文件夹内
# Important : Please make sure your files are saved to the 'results' folder
# in your jobs
###############################################################################
import warnings
# 忽视警告
warnings.filterwarnings('ignore')

import cv2
from PIL import Image
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch_py.Utils import plot_image
from torch_py.MTCNN.detector import FaceDetector
from torch_py.MobileNetV1 import MobileNetV1
from torch_py.FaceRec import Recognition
# 数据集路径
data_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/"

def processing_data(data_path, height=224, width=224, batch_size=32,
                    test_split=0.2):
    """
    数据处理部分
    :param data_path: 数据路径
    :param height:高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return: 
    """
    transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        T.ToTensor(),  # 转化为张量
        T.Normalize([0.5], [0.25]),  # 归一化
    ])

    dataset = ImageFolder(data_path, transform=transforms)
    # 划分数据集
    train_size = int((1-test_split)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # 创建一个 DataLoader 对象
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
    valid_data_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    return train_data_loader, valid_data_loader


data_path = './datasets/5f680a696ec9b83bb0037081-momodel/data/image'
# train_data_loader, valid_data_loader = processing_data(data_path=data_path, height=160, width=160, batch_size=32)
# pnet_path = "./torch_py/MTCNN/weights/pnet.npy"
# rnet_path = "./torch_py/MTCNN/weights/rnet.npy"
# onet_path = "./torch_py/MTCNN/weights/onet.npy"
# torch.set_num_threads(1)

# # 读取测试图片
# img = Image.open("test.jpg")

# # 加载模型进行识别口罩并绘制方框
# recognize = Recognition()
# draw = recognize.face_recognize(img)
# plot_image(draw)
# 加载 MobileNet 的预训练模型权
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

train_data_loader, valid_data_loader = processing_data(data_path=data_path, height=160, width=160, batch_size=32)
# modify_x, modify_y = torch.ones((32, 3, 160, 160)), torch.ones((32))

# epochs = 20
epochs = 300
model = MobileNetV1(classes=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay = 1e-6)  # 优化器
# optimizer = optim.Adam(model.parameters(), lr=1e-5)
print('加载完成...')

# 学习率下降的方式，acc三次不下降就下降学习率继续训练，衰减学习率
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                  'max', 
#                                                  factor=0.5,
#                                                  patience=2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 'max', 
                                                 factor=0.1,
                                                 patience=4)
# 损失函数
criterion = nn.CrossEntropyLoss()  
best_loss = 1e9
best_model_weights = copy.deepcopy(model.state_dict())
# loss_list = []  # 存储损失函数值
for epoch in range(epochs):
    model.train()
    l = 0
    i = 0
    for batch_idx, (x, y) in enumerate(train_data_loader, 1):
        x = x.to(device)
        y = y.to(device)
        pred_y = model(x)

        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l += loss
        i += 1
        avg_loss = l*1.0/i

        if loss < best_loss:
            best_model_weights = copy.deepcopy(model.state_dict())
            best_loss = loss
            
    if epoch > 90 and best_loss < 0.01 and avg_loss < 0.02:
        torch.save(model.state_dict(), './results/temp'+str(epoch+1)+'.pth')

    print('step:' + str(epoch + 1) + '/' + str(epochs) + ' || best Loss: %.4f' % (loss) + '|| avg loss:%.4f'%(l*1.0/i))

torch.save(model.state_dict(), './results/temp.pth')
print('Finish Training.')

img = Image.open("test.jpg")
img1 = Image.open("test1.jpg")
detector = FaceDetector()
recognize = Recognition(model_path='results/temp.pth')
_ , all_num, mask_nums = recognize.mask_recognize(img)
print("all_num:", all_num, "mask_num", mask_nums)
_ , all_num, mask_nums = recognize.mask_recognize(img1)
print("all_num:", all_num, "mask_num", mask_nums)

