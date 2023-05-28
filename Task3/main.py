import os
import torch 
import torch.nn as nn
import cv2
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
from torchvision.models.resnet import resnet50


def predict(img):
    """
    加载模型和模型预测
    主要步骤:
        1.图片处理
        2.用加载的模型预测图片的类别
    :param img: 经 cv2.imread(file_path) 读取后的图片
    :return: string, 模型识别图片的类别, 
            共 'CL', 'FBB', 'HG', 'HJ', 'LHR', 'LSS', 'LYF', 'PYY', 'TY', 'YM' 10 个类别
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------

    model = resnet50(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs,256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10)
    )
    model.load_state_dict(torch.load('./results/resnet/net_91.pth.tar', map_location='cpu'))

    # # 图片放缩
    # img = cv2.resize(img, dsize=(90, 90), interpolation=cv2.INTER_CUBIC)
    # img = 1.0/255 * img
    # # expand_dims的作用是把img.shape转换成(1, img.shape[0], img.shape[1], img.shape[2])
    # x = torch.from_numpy(np.expand_dims(img, axis=3))

    t = transforms.Compose([
            transforms.Resize((120, 120)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.25]),
            ])

    if isinstance(img, np.ndarray):
        # 转化为 PIL.JpegImagePlugin.JpegImageFile 类型
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    img = t(img)
    # img = np.array(img)
    # img = img * 1.0 / 255
    # img = transforms.ToTensor()(img).unsqueeze(0)
    img = img.unsqueeze(0)
    img = torch.tensor(img, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        # predict_label = model(img).data.numpy()
        predict_label = model(img)
        # current_class = self.classes[np.argmax(predict_label).item()]
        y = np.argmax(predict_label).item()
   
    # 获取labels
    labels = {0: 'CL', 1: 'FBB', 2: 'HG', 3: 'HJ', 4: 'LHR', 5: 'LSS', 6: 'LYF', 7: 'PYY', 8: 'TY', 9: 'YM'}
    # 获取输入图片的类别
    y_predict = labels[y]

    # -------------------------------------------------------------------------
    
    # 返回图片的类别
    return y_predict