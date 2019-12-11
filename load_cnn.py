# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 2:25 下午
# @Author  : Huang Hongbing
# @File    : load_cnn.py
# @Software: PyCharmfrom
from keras.models import model_from_json
import numpy as np
import cv2
import pandas as pd


def cnn_predict(path):
    #1.加载已经训练好的cnn模型
    cnn_model = model_from_json(open('./action_recognize/model/cnn_model_architecture.json').read())
    cnn_model.load_weights('./action_recognize/model/cnn_model_weights.h5')
    #print(cnn_model.summary()) 查看模型参数等信息

    #2.加载并处理图片成为模型的输入
    image_test= path    #选择图片
    img_open = cv2.imread(image_test,cv2.IMREAD_GRAYSCALE)  # 打开图像
    img_ndarray = np.asarray(img_open, dtype='float64') / 256  # 将图像转化为数组并将像素转化到0-1之间
    image_data = cv2.resize(img_ndarray,(784,1)) #将图片剪切成784个像素
    image_dataframe = pd.DataFrame(image_data) #将图片的784个像素数据存入dataframe
    #print(image_dataframe.shape)  output:(1, 784)
    image_train = image_dataframe.values.reshape(-1,28,28,1)
    #print(X_train.shape) output:(1, 28, 28, 1)

    #labels=["stand","wave","flap","squat","bowling"] 对应的动作类别,实际预测结果显示的将是1-5
    #3.预测图片类别
    image_predict = cnn_model.predict(image_train)
    image_predict = np.argmax(image_predict,axis = 1)
    # print(image_predict)
    return image_predict
