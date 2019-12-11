# -*- coding: utf-8 -*-
# @Time    : 2019/12/2 16:34 下午
# @Author  : Zhou Liang
# @File    : predict.py
# @Software: PyCharm

import time
import sys
import glob
import os
import argparse
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from sklearn import preprocessing
from keras.models import load_model  #　keras导入模型
from sklearn.externals import joblib  # 导出/导入sklearn模型
import get_keypoints_position # 获取关节位置
# 导入cnn模型
import load_cnn
# tf-open-pose
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


# 定义动作类型
action_type = {
    0: 'Stand',
    1: 'Wave',
    2: 'Flap',
    3: 'Squat',
    4: 'Bowling',
}


def submit(text):
    ydata = eval(text)
    l.set_ydata(ydata)
    ax.set_ylim(np.min(ydata), np.max(ydata))
    plt.draw()

def recognize_action_from_image(image_path, model='dnn_model'):
    start_time = time.time()
    # 导入模型
    model = load_model('./action_recognize/model/' + 'dnn_model.h5')
    # 训练one-hot-encoding编码器
    df = pd.read_csv('./csv/output_normalized_cmu.csv', header=0)
    X = df.loc[:, :'17_y'].astype(float)
    Y = df.loc[:, ['type_index']]
    encoder = preprocessing.LabelBinarizer()
    dummy_Y = encoder.fit_transform(Y)
    # 获得图片中的关节位置列表
    all_keypoints_list = get_keypoints_position.get_keypoints_from_image(image_path)
    for idx, person in enumerate(all_keypoints_list):
        x = np.array(all_keypoints_list[0]).reshape((1, 36))
        pred_action = model.predict(x)
        action_idx = encoder.inverse_transform(pred_action)[0]
        print('图片中第%d个人的关节位置为:' % (idx + 1), all_keypoints_list[idx], ',其动作为:', action_type[action_idx])
    elapsed_time = time.time() - start_time
    print('耗时%.2fseconds' % elapsed_time)


if __name__ == '__main__':
    # 对一个目录下的所有图片进行动作识别
    parser = argparse.ArgumentParser(description='调用模型对图片进行动作识别')
    parser.add_argument('--mode', type=str, default='dir', help='执行方式dir识别目录 / file识别单文件')
    parser.add_argument('--path', type=str, default='./images/recognize', help='要识别的目录/图片的路径')
    parser.add_argument('--model', type=str, default='mobilenet_v2_large',
                        help='tf-open-pose调用的与训练模型,可选cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='432x368',
                        help='姿态估计前缩放图片'
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')

    args = parser.parse_args()
    start_time = time.time()
    # 导入模型
    try:
        dnn_model = load_model('./action_recognize/model/' + 'dnn_model.h5')
        svm_model = joblib.load("./action_recognize/model/svm_model.m")
        naivebayes_model = joblib.load("./action_recognize/model/naivebayes_model.m")
        knn_model = joblib.load("./action_recognize/model/knn_model.m")
        rfc_model = joblib.load("./action_recognize/model/rfc_model.m")
        decision_tree_model = joblib.load("./action_recognize/model/decision_tree_model.m")
    except:
        print('模型载入失败！')
    else:
        print('模型载入成功.')
    # 训练one-hot-encoding编码器
    df = pd.read_csv('./csv/output_normalized_cmu.csv', header=0)
    X = df.loc[:, :'17_y'].astype(float)
    Y = df.loc[:, ['type_index']]
    encoder = preprocessing.LabelBinarizer()
    dummy_Y = encoder.fit_transform(Y)
    # tf-open-pose
    w, h = model_wh(args.resize)
    estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    if args.mode == 'dir':  # 识别目录下所有图片
        files_grabbed = glob.glob(os.path.join(args.path, '*.*'))
        all_count = len(files_grabbed)
        count = 0
        for i, file in enumerate(files_grabbed):
            bodys_in_file = []  # 一张图片中可能有多人
            image = common.read_imgfile(file, None, None)
            humans = estimator.inference(image, True, 4.0)
            estimation_results = get_keypoints_position.get_keypoints_from_humans(humans)
            count += 1
            print('-'*50)
            print('[%d/%d]识别图片%s' % (count, all_count, file))
            if len(estimation_results) == 0:
                print('\t无法从图片中识别出人物！')
                continue
            for item in estimation_results:
                bodys_in_file.append(item)
            for idx, person in enumerate(bodys_in_file):
                print('\t' + '图中第%d个人的关节位置为:' % (idx+1), bodys_in_file[idx])
                x = np.array(bodys_in_file[idx]).reshape((1, 36))
                dnn_pred = encoder.inverse_transform(dnn_model.predict(x))
                print(dnn_pred)
                print('\t\t', 'DNN模型识别结果:', action_type[dnn_pred[0]])
                knn_pred = knn_model.predict(x)
                print('\t\t', 'KNN模型识别结果:', action_type[knn_pred[0]])
                rfc_pred = rfc_model.predict(x)
                print('\t\t', 'RFC模型识别结果:', action_type[rfc_pred[0]])
                svm_pred = svm_model.predict(x)
                print('\t\t', 'SVM模型识别结果:', action_type[svm_pred[0]])
                naivebayes_pred = naivebayes_model.predict(x)
                print('\t\t', 'NaiveBayes模型识别结果:', action_type[naivebayes_pred[0]])
                decision_tree_pred = encoder.inverse_transform(decision_tree_model.predict(x))
                print('\t\t', 'DecisionTree模型识别结果:', action_type[decision_tree_pred[0]])
        elapsed_time = time.time() - start_time
        print('-' * 50)
        print('共耗时%.2fseconds' % elapsed_time)
    else: # 识别单张图片
        estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
        image = common.read_imgfile(args.path, None, None)
        humans = estimator.inference(image, True, 4.0)
        image_draw = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        estimation_results = get_keypoints_position.get_keypoints_from_humans(humans)
        recognize_results = ''
        print('识别图片%s' % (args.path))
        if len(estimation_results) == 0:
            print('\t无法从图片中识别出人物！')
        else:
            for idx, person in enumerate(estimation_results):
                print('-' * 50)
                print('\t' + '图中第%d个人的关节位置为:' % (idx+1), estimation_results[idx])
                recognize_results += 'Person%d:    ' % (idx+1)
                x = np.array(estimation_results[idx]).reshape((1, 36))
                dnn_pred = encoder.inverse_transform(dnn_model.predict(x))
                recognize_results += '[DNN]:%s    ' % (action_type[dnn_pred[0]])
                print('\t\t', 'DNN模型识别结果:', action_type[dnn_pred[0]])
                knn_pred = knn_model.predict(x)
                recognize_results += '[KNN]:%s    ' % (action_type[knn_pred[0]])
                print('\t\t', 'KNN模型识别结果:', action_type[knn_pred[0]])
                rfc_pred = rfc_model.predict(x)
                recognize_results += '[RFC]:%s    ' % (action_type[rfc_pred[0]])
                print('\t\t', 'RFC模型识别结果:', action_type[rfc_pred[0]])
                svm_pred = svm_model.predict(x)
                recognize_results += '[SVM]:%s    ' % (action_type[svm_pred[0]])
                print('\t\t', 'SVM模型识别结果:', action_type[svm_pred[0]])
                naivebayes_pred = naivebayes_model.predict(x)
                recognize_results += '[NaiveBayes]:%s    ' % (action_type[naivebayes_pred[0]])
                print('\t\t', 'NaiveBayes模型识别结果:', action_type[naivebayes_pred[0]])
                decision_tree_pred = encoder.inverse_transform(decision_tree_model.predict(x))
                recognize_results += '[DecisionTree]:%s    ' % (action_type[decision_tree_pred[0]])
                print('\t\t', 'DecisionTree模型识别结果:', action_type[decision_tree_pred[0]])
                cnn_pred = load_cnn.cnn_predict(args.path)
                recognize_results += '[CNN]:%s    ' % (action_type[cnn_pred[0]-1])
                print('\t\t', 'CNN模型识别结果:', action_type[cnn_pred[0]-1])
                recognize_results += '\n'
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.figure()
        # plt.imshow(image_draw)
        # plt.title('Recognition Result')
        # plt.show()
        elapsed_time = time.time() - start_time
        print('-' * 50)
        print('共耗时%.2fseconds' % elapsed_time)
        fig, ax = plt.subplots(figsize=(30,30))
        plt.subplots_adjust(bottom=0.2)
        plt.imshow(image_draw)
        plt.title('Pose Estimation & Action Recognization', fontsize=20)
        initial_text = recognize_results
        axbox = plt.axes([0.1, 0.05, 0.8, 0.125])
        text_box = TextBox(axbox, 'Result:', initial=initial_text)
        text_box.on_submit(submit)
        plt.show()



