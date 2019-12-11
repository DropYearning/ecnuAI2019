# -*- coding: utf-8 -*-
# @Time    : 2019/11/29 3:15 下午
# @Author  : Zhou Liang
# @File    : normalization.py
# @Software: PyCharm

import argparse
import time
import csv


def normalization_from_list(person_keypoints):
    normalized_result = []
    x_max = 0
    x_min = 1
    y_max = 0
    y_min = 1
    for idx, bodypart_pos in enumerate(person_keypoints):
        if idx % 2 == 0:  # x
            x = person_keypoints[idx]
            x_max = x if x > x_max else x_max
            x_min = x if (x < x_min and x != 0) else x_min
        elif idx % 2 == 1:  # y
            y = person_keypoints[idx]
            y_max = y if y > y_max else y_max
            y_min = y if (y < y_min and y != 0) else y_min
    for idx, bodypart_pos in enumerate(person_keypoints):
        if idx % 2 == 0:  # x
            x = person_keypoints[idx]
            if x != 0:  # 不对未检测到的关节做归一化
                x_normalized = (x - x_min) / (x_max - x_min)
                normalized_result.append(round(x_normalized, 2))
            else:
                normalized_result.append(x)
        elif idx % 2 == 1:  # y
            y = person_keypoints[idx]
            if y != 0:  # 不对未检测到的关节做归一化
                y_normalized = (y - y_min) / (y_max - y_min)
                normalized_result.append(round(y_normalized, 2))
            else:
                normalized_result.append(y)
    return normalized_result

if __name__ == '__main__':
    # 创建结果csv文件， 将整个csv文件中的数据归一化
    csv_file = open('output_normalized.csv', 'w')
    csv_write = csv.writer(csv_file)
    csv_head = [
        "0_x", "0_y",
        "1_x", "1_y",
        "2_x", "2_y",
        "3_x", "3_y",
        "4_x", "4_y",
        "5_x", "5_y",
        "6_x", "6_y",
        "7_x", "7_y",
        "8_x", "8_y",
        "9_x", "9_y",
        "10_x", "10_y",
        "11_x", "11_y",
        "12_x", "12_y",
        "13_x", "13_y",
        "14_x", "14_y",
        "15_x", "15_y",
        "16_x", "16_y",
        "17_x", "17_y",
        'type_index', 'type_str', 'file_name'
    ]
    csv_write.writerow(csv_head)
    csv_file.close()
    parser = argparse.ArgumentParser(description='归一化')
    parser.add_argument('--input', type=str, default='output.csv')
    args = parser.parse_args()
    time_start = time.time()
    csv_reader = csv.reader(open(args.input))
    csv_file = open('output_normalized.csv', 'a')
    csv_write = csv.writer(csv_file)
    for i, row in enumerate(csv_reader):
        if i != 0:
            x_max = 0
            x_min = 1
            y_max = 0
            y_min = 1
            input_row = row
            output_row = []
            for index, item in enumerate(input_row):
                if index < 54 and index % 3 == 0:  # x
                    x = float(input_row[index])
                    x_max = x if x > x_max else x_max
                    x_min = x if (x < x_min and x != 0) else x_min
                elif index < 54 and index % 3 == 1:  # y
                    y = float(input_row[index])
                    y_max = y if y > y_max else y_max
                    y_min = y if (y < y_min and y != 0) else y_min
            for index, item in enumerate(input_row):
                if index < 54:
                    if index % 3 == 0:  # x
                        x = float(input_row[index])
                        x_normalized = (x - x_min) / (x_max - x_min)
                        if x != 0:   # 不对未检测到的关节做归一化
                            output_row.append(round(x_normalized, 2))
                        else:
                            output_row.append(x)
                    elif index % 3 == 1:  # y
                        y = float(input_row[index])
                        y_normalized = (y - y_min) / (y_max - y_min)
                        if y !=0:  # 不对未检测到的关节做归一化
                            output_row.append(round(y_normalized, 2))
                        else:
                            output_row.append(y)
            output_row.append(input_row[54])
            output_row.append(input_row[55])
            output_row.append(input_row[56])
            csv_write.writerow(output_row)
            print(i, 'x_min:', x_min, ',x_max', x_max, ',y_min', y_min, ',y_max', y_max)
    csv_file.close()
    time_elapsed = time.time() - time_start
    print("Normalization finished in %.2f seconds" % time_elapsed)
