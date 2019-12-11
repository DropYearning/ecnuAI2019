# -*- coding: utf-8 -*-
# @Time    : 2019/12/2 2:34 下午
# @Author  : Zhou Liang
# @File    : get_keypoints_position.py
# @Software: PyCharm
import sys
import argparse
import glob
import os
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from normalization import normalization_from_list



def get_keypoints_from_image(image_path, model='mobilenet_v2_large', resolution='432x368', normalizetion=True):
    all_keypoints = []  # 返回一个关节位置序列的列表（一张图片中可能有多人）
    w, h = model_wh(resolution)
    estimator = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    image = common.read_imgfile(image_path, None, None)
    humans = estimator.inference(image, True, 4.0)
    for human in humans:
        body_dict = dict()
        human_split = str(human).split('BodyPart:')
        for bodypart in human_split:
            left_parenthesis = bodypart.find('(')
            if left_parenthesis != -1:
                if left_parenthesis == 2:
                    bodypart_index = int(bodypart[left_parenthesis - 2:left_parenthesis - 1])
                else:
                    bodypart_index = int(bodypart[left_parenthesis - 3:left_parenthesis - 1])
                x_pos = float(bodypart[left_parenthesis + 1:left_parenthesis + 5])
                y_pos = float(bodypart[left_parenthesis + 7:left_parenthesis + 11])
                confidence_score = float(bodypart[left_parenthesis + 19:left_parenthesis + 23])
                body_dict[bodypart_index] = [x_pos, y_pos, confidence_score]
            else:
                continue
        # 缺失的关节点补0
        for q in range(18):
            if q not in body_dict.keys():
                body_dict[q] = [0, 0, 0]
        keypoints = [
            body_dict[0][0], body_dict[0][1],
            body_dict[1][0], body_dict[1][1],
            body_dict[2][0], body_dict[2][1],
            body_dict[3][0], body_dict[3][1],
            body_dict[4][0], body_dict[4][1],
            body_dict[5][0], body_dict[5][1],
            body_dict[6][0], body_dict[6][1],
            body_dict[7][0], body_dict[7][1],
            body_dict[8][0], body_dict[8][1],
            body_dict[9][0], body_dict[9][1],
            body_dict[10][0], body_dict[10][1],
            body_dict[11][0], body_dict[11][1],
            body_dict[12][0], body_dict[12][1],
            body_dict[13][0], body_dict[13][1],
            body_dict[14][0], body_dict[14][1],
            body_dict[15][0], body_dict[15][1],
            body_dict[16][0], body_dict[16][1],
            body_dict[17][0], body_dict[17][1],
        ]
        if not normalizetion:
            all_keypoints.append(keypoints)  # 不归一化
        else:
            all_keypoints.append(normalization_from_list(keypoints))  # 归一化
    return all_keypoints


# 从tf-openpose输出的humans数据结构中返回图中所有人物的关节位置序列
def get_keypoints_from_humans(humans, normalizetion=True):
    all_keypoints = []  # 返回一个关节位置序列的列表（一张图片中可能有多人）
    for human in humans:
        body_dict = dict()
        human_split = str(human).split('BodyPart:')
        for bodypart in human_split:
            left_parenthesis = bodypart.find('(')
            if left_parenthesis != -1:
                if left_parenthesis == 2:
                    bodypart_index = int(bodypart[left_parenthesis - 2:left_parenthesis - 1])
                else:
                    bodypart_index = int(bodypart[left_parenthesis - 3:left_parenthesis - 1])
                x_pos = float(bodypart[left_parenthesis + 1:left_parenthesis + 5])
                y_pos = float(bodypart[left_parenthesis + 7:left_parenthesis + 11])
                confidence_score = float(bodypart[left_parenthesis + 19:left_parenthesis + 23])
                body_dict[bodypart_index] = [x_pos, y_pos, confidence_score]
            else:
                continue
        # 缺失的关节点补0
        for q in range(18):
            if q not in body_dict.keys():
                body_dict[q] = [0, 0, 0]
        keypoints = [
            body_dict[0][0], body_dict[0][1],
            body_dict[1][0], body_dict[1][1],
            body_dict[2][0], body_dict[2][1],
            body_dict[3][0], body_dict[3][1],
            body_dict[4][0], body_dict[4][1],
            body_dict[5][0], body_dict[5][1],
            body_dict[6][0], body_dict[6][1],
            body_dict[7][0], body_dict[7][1],
            body_dict[8][0], body_dict[8][1],
            body_dict[9][0], body_dict[9][1],
            body_dict[10][0], body_dict[10][1],
            body_dict[11][0], body_dict[11][1],
            body_dict[12][0], body_dict[12][1],
            body_dict[13][0], body_dict[13][1],
            body_dict[14][0], body_dict[14][1],
            body_dict[15][0], body_dict[15][1],
            body_dict[16][0], body_dict[16][1],
            body_dict[17][0], body_dict[17][1],
        ]
        if not normalizetion:
            all_keypoints.append(keypoints)  # 不归一化
        else:
            all_keypoints.append(normalization_from_list(keypoints))  # 归一化
    return all_keypoints


def get_keypoints_from_directory(direction_path, format='jpg', model='mobilenet_v2_large', resolution='432x368'):
    all_keypoints = []  # 返回一个关节位置序列的列表（一张图片中可能有多人）
    w, h = model_wh(resolution)
    estimator = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    files_grabbed = glob.glob(os.path.join(direction_path, '*.' + format))
    for i, file in enumerate(files_grabbed):
        image = common.read_imgfile(file, None, None)
        humans = estimator.inference(image, True, 4.0)
        results = get_keypoints_from_humans(humans)
        for item in results:
            all_keypoints.append(item)
    return all_keypoints

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从图片中得到所有人的关节位置序列的列表')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='mobilenet_v2_large',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    args = parser.parse_args()
    results = get_keypoints_from_image(args.image, model=args.model, resolution=args.resize)
    for idx, person in enumerate(results):
        print('第%d个人:' % (idx+1), results[idx])
