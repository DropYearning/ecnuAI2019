# 作者：马言悦
import pandas as pd
import numpy as np
#1.# 读取训练数据
from scipy.interpolate._ppoly import evaluate
from sklearn.ensemble import RandomForestClassifier
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
# 3.特征工程(对特征值进行标准化处理)
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)
 # 4.送入算法
rfc = RandomForestClassifier() # 创建一个RFC算法实例
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rfc.get_params())
#为了使用RandomizedSearchCV，我们首先需要创建一个参数网格在拟合过程中进行采样
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)
rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rfc_random.fit(x_train, y_train.values.ravel())
rfc_random.best_params_ #输出最佳参数
rfc_random.best_estimator_#输出最佳模型
rfc_random.best_score_#输出最佳评估


#重构模型
rfc=rfc_random.best_estimator_
rfc.fit(x_train,y_train.values.ravel())
y_predict=rfc.predict(x_test)
print("准确率：",rfc.score(x_test, y_test))
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,y_predict,digits=4))
joblib.dump(rfc, "rfc_model.m")
import matplotlib.pyplot as plt
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

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

labels = ["stand", "wave", "flap","squat","bowling"]
plot_Matrix(cm, labels, "RFC Confusion Matrix")