# -*- coding: utf-8 -*-
# @Time    : 2019/12/3 4:34 下午
# @Author  : Zhou Liang
# @File    : svm_train.py
# @Software: PyCharm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt  # plot
from sklearn.model_selection import train_test_split # 划分数据集
from sklearn import svm  # SVM
from sklearn.externals import joblib  # 导出sklearn模型
from sklearn import metrics # 评估模型
from sklearn.metrics import confusion_matrix # 混淆矩阵

# 绘制混淆矩阵
def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='8')  # 设置字体样式、大小

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('cm.png', dpi=300)
    plt.show()

#　查看是否在使用gpu进行训练
print(tf.test.is_gpu_available())
# 准备数据
df = pd.read_csv('output_normalized_cmu.csv', header=0)
seed = 42
# x为36维特征
x = df.loc[:, :'17_y'].astype(float)
# y为动作种类（标签）
y = df.loc[:,['type_index'] ]
labels_name = ['Stand', 'Wave', 'Flap', 'Squat', 'Bowling']
# 划分数据集
x_temp,x_test,y_temp,y_test = train_test_split(x, y, test_size=0.2, random_state=seed) # 第一次划分，划分出20%为测试集：x_test, y_test
x_train, x_validation, y_train, y_validation = train_test_split(x_temp, y_temp, test_size=0.25, random_state=seed) # 第二次划分，划分出60%的训练集和20%的验证集
print('训练集大小:', len(x_train),  '验证集大小:', len(x_validation),  '测试集大小:', len(x_test))
# 初始化SVM
svc = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)

nusvc = svm.NuSVC(cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
      kernel='rbf', max_iter=-1, nu=0.5, probability=False, random_state=None,
      shrinking=True, tol=0.001, verbose=False)

linear_svc = svm.LinearSVC(C=250.0, dual=True, fit_intercept=True,
          loss='squared_hinge',multi_class='ovr', penalty='l2', random_state=seed, tol=0.0001,)
# 在训练集上训练
svc.fit(x_train,y_train)
nusvc.fit(x_train,y_train)
linear_svc.fit(x_train,y_train)
# 在验证集上调整模型
print("SVC Evaluation on validation data: accuracy = %0.3f%% \n"  % ( svc.score(x_validation, y_validation) * 100) )
print("NuSVC Evaluation on validation data: accuracy = %0.3f%% \n"  % ( nusvc.score(x_validation, y_validation) * 100) )
print("Linear_SVC Evaluation on validation data: accuracy = %0.3f%% \n"  % ( linear_svc.score(x_validation, y_validation) * 100) )
# 在测试集评估模型
y_pred = linear_svc.predict(x_test)
y_true = y_test
print(metrics.classification_report(y_test, y_pred, digits=4, target_names=labels_name))
# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)
plot_Matrix(cm, labels_name, "SVM Confusion Matrix")
# 导出sklearn模型
joblib.dump(svc, "svm_model.m")
# 对一组数据进行预测
input = [0.42, 0.0, 0.33, 0.14, 0.17, 0.14, 0.08, 0.31, 0.0, 0.48, 0.5, 0.14, 0.75, 0.1, 1.0, 0.03, 0.25, 0.48, 0.25, 0.76, 0.17, 1.0, 0.42, 0.48, 0.42, 0.76, 0.42, 1.0, 0.33, 0.0, 0.42, 0.0, 0.25, 0.0, 0.42, 0.03]
input = np.array(input).reshape((1, 36))
pred_output = svc.predict(input)
print(pred_output)