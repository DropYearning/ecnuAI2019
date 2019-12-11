# -*- coding: utf-8 -*-
# @Time    : 2019/11/27 5:10 下午
# @Author  : Zhou Liang
# @File    : run_directory.py
# @Software: PyCharm
import argparse
import logging
import time
import glob
import os
import sys
import csv

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

input_action_type = ['stand', 'wave', 'flap', 'squat', 'bowling']

# 创建结果csv文件
csv_file = open('output.csv', 'w')
csv_write = csv.writer(csv_file)
csv_head = [
            "0_x", "0_y", "0_score",
            "1_x", "1_y", "1_score",
            "2_x", "2_y", "2_score",
            "3_x", "3_y", "3_score",
            "4_x", "4_y", "4_score",
            "5_x", "5_y", "5_score",
            "6_x", "6_y", "6_score",
            "7_x", "7_y", "7_score",
            "8_x", "8_y", "8_score",
            "9_x", "9_y", "9_score",
            "10_x", "10_y", "10_score",
            "11_x", "11_y", "11_score",
            "12_x", "12_y", "12_score",
            "13_x", "13_y", "13_score",
            "14_x", "14_y", "14_score",
            "15_x", "15_y", "15_score",
            "16_x", "16_y", "16_score",
            "17_x", "17_y", "17_score",
            'type_index', 'type_str', 'file_name'
            ]
csv_write.writerow(csv_head)
csv_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf_pose_estimation run by folder')
    # parser.add_argument('--folder', type=str, default='./input/')
    parser.add_argument('--format', type=str, default='jpg', help='jpg / png / ...')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_v2_large', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    # parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()

    csv_file = open('output.csv', 'a')
    csv_write = csv.writer(csv_file)
    w, h = model_wh(args.resolution)
    images_format = args.format
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    time_all_start = time.time()
    all_file_count = 0
    all_body_count = 0
    empty_body_file_list = []
    extra_body_file_list = []
    for action_index, action in enumerate(input_action_type):
        direction_path = r"./input/" + action + r'/'
        files_grabbed = glob.glob(os.path.join(direction_path, '*.' + images_format))
        files_count = len(files_grabbed)
        all_file_count += files_count
        body_count = 0
        # if files_count == 0:
        #     logger.error("The directory has no image in type" + images_format + '!')
        #     sys.exit(-1)
        for i, file in enumerate(files_grabbed):
            image = common.read_imgfile(file, None, None)
            time_file_start = time.time()
            humans = e.inference(image, True, 4.0)
            time_file_elapsed = time.time() - time_file_start
            # logger.info('Processed image #%d/%d:  %s in %.4f seconds.' % (i, i, file, time_file_elapsed))
            # 图片的背景中有三脚架有可能会被openpose识别为一个身体
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
                body_dict['filename'] = file
                body_dict['type_index'] = action_index
                body_dict['type_name'] = action
                # print(body_dict)
                data_row = [
                            body_dict[0][0], body_dict[0][1], body_dict[0][2],
                            body_dict[1][0], body_dict[1][1], body_dict[1][2],
                            body_dict[2][0], body_dict[2][1], body_dict[2][2],
                            body_dict[3][0], body_dict[3][1], body_dict[3][2],
                            body_dict[4][0], body_dict[4][1], body_dict[4][2],
                            body_dict[5][0], body_dict[5][1], body_dict[5][2],
                            body_dict[6][0], body_dict[6][1], body_dict[6][2],
                            body_dict[7][0], body_dict[7][1], body_dict[7][2],
                            body_dict[8][0], body_dict[8][1], body_dict[8][2],
                            body_dict[9][0], body_dict[9][1], body_dict[9][2],
                            body_dict[10][0], body_dict[10][1], body_dict[10][2],
                            body_dict[11][0], body_dict[11][1], body_dict[11][2],
                            body_dict[12][0], body_dict[12][1], body_dict[12][2],
                            body_dict[13][0], body_dict[13][1], body_dict[13][2],
                            body_dict[14][0], body_dict[14][1], body_dict[14][2],
                            body_dict[15][0], body_dict[15][1], body_dict[15][2],
                            body_dict[16][0], body_dict[16][1], body_dict[16][2],
                            body_dict[17][0], body_dict[17][1], body_dict[17][2],
                            body_dict['type_index'], body_dict['type_name'], body_dict['filename']
                            ]
                csv_write.writerow(data_row)
            print('[%d / %d] %s type image %s has been processed in %.2f seconds.' % (i+1, files_count, action, file, time_file_elapsed ))
            body_count += len(humans)
            if len(humans) == 0:
                empty_body_file_list.append(file)
            elif len(humans) > 1:
                extra_body_file_list.append(file)
        all_body_count += body_count
    time_all_elapsed = time.time() - time_all_start
    csv_file.close()
    # logger.info('All images finished in %.4f seconds.' % time_all_elapsed)
    print('All images finished in %.4f seconds.' % time_all_elapsed)
    print("all input files count: ", all_file_count)  # 总的输入图片数
    print("body count in all input files: ", all_body_count)  # 总的输入图片中的人数
    print("These %d images have zero body detected:" % len(empty_body_file_list))  # 输出没识别出身体的图片名
    for file in empty_body_file_list:
        print(file)
    print("")
    print("These %d images have more than one body detected:" % len(extra_body_file_list))  # 输出识别出的身体数量超过1个的图片名
    for file in extra_body_file_list:
        print(file)


