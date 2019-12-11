# -*- coding: utf-8 -*-
# @Time    : 2019/12/5 7:11下午
# @Author  : Huang Hongbing
# @File    : knn_train.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#1.# 读取训练数据
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

csv_data = pd.read_csv('output_normalized_cmu.csv')
#print(csv_data.shape)  # (2195, 39)
csv_coordinate_data = csv_data.iloc[:, 0:36]  # 取前面36条数据作为关节坐标数据数据
#print(csv_coordinate_data.shape)  # (2195, 36)
csv_tag_data = csv_data.iloc[:, [36]]  # 取第36条数据作为标签数据
#print(csv_tag_data.shape)  # (2195, 1)

#2.划分数据集
from sklearn.model_selection import train_test_split
#设置分割率为20%
x_train, x_test, y_train, y_test = train_test_split(csv_coordinate_data, csv_tag_data, test_size=0.25, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

 # 4.送入算法
knn = KNeighborsClassifier(n_neighbors=5) # 创建一个KNN算法实例，n_neighbors默认为5,后续通过网格搜索获取最优参数
knn.fit(x_train, y_train.values.ravel()) # 将测试集送入算法
y_predict = knn.predict(x_test) # 获取预测结果
y_predict=y_predict.T
y_test=y_test.T
print(y_test.iloc[0,6])
labels = ["stand", "wave", "flap","squat","bowling"]
#for i in range(len(y_predict)):
    #print("第%d次测试:真实值:%s\t预测值:%s"%((i+1),labels[y_predict[i]],labels[y_test.iloc[0,i]]))
print("准确率：",knn.score(x_test, y_test.T))
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test.T,y_predict.T,target_names=labels,digits=4))
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


cm = confusion_matrix(y_test.T, y_predict.T)
plot_Matrix(cm, labels, "KNN Confusion Matrix")
#=======寻找到最好的k===========

best_score=0.0  #先设置一个精确度的初始值
best_k=-1         #设置一个k的初始值
#综合考虑距离和不考虑距离来选取最优的k值
best_methon=''
best_score=0.0
best_k=-1
for methon in ['uniform','distance']:
    for k in range(3,11):
        knn_clf=KNeighborsClassifier(n_neighbors=k,weights=methon)
        knn_clf.fit(x_train,y_train.values.ravel())
        score=knn_clf.score(x_test,y_test.T)
        if score > best_score:
            best_score=score
            best_k=k
            best_methon=methon
print('best_methon=%s'%(best_methon))
print('best_k=%s'%(best_k))
print('best_score=%s'%(best_score))
print("开始网格搜索最佳参数\n\n")
#===网格搜索=====

#Grid Search

param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors':[i for i in range(1,11)]
    },
    {
        'weights':['distance'],
        'n_neighbors':[i for i in range(1,11)],
        'p':[i for i in range(1,6)]
    }
]
knn_clf=KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV  #导入相应的网格搜素模块
grid_search=GridSearchCV(knn_clf,param_grid) #这个相当于代替之前的两个for循环
grid_search.fit(x_train,y_train.values.ravel()) #传入相应的参数进行拟合
'''
grid_search.best_estimator_  #这个参数可以知道每个参数的最优选择
grid_search.best_score_  #这个参数可以知道该算法的精确度
grid_search.best_params_  #这个参数可以知道最优的参数选择方案
'''
#===============
#得到上述参数后，我们就重新定义一个分类器
knn_clf=grid_search.best_estimator_
knn_clf.fit(x_train,y_train.values.ravel())
y_predict=knn_clf.predict(x_test)
print("准确率：",knn_clf.score(x_test,y_test.T)) #算法的精确度  0.9833333333333333
from sklearn.metrics import classification_report
print(classification_report(y_test.T,y_predict,digits=4))
joblib.dump(knn_clf, "knn_model.m")
