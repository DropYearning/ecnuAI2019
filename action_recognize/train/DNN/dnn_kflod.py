# -*- coding: utf-8 -*-
# @Time    : 2019/12/2 4:14 下午
# @Author  : Zhou Liang
# @File    : dnn_kflod.py
# @Software: PyCharm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.utils import np_utils
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder  # label process
from keras.models import Sequential
from keras.layers import Dense, Dropout

# 准备数据
df = pd.read_csv('output_normalized_cmu.csv', header=0)
df_shuffled = df.sample(frac=1)
seed = 42
# x为36维特征
X = df_shuffled.iloc[:, :36].astype(float)
# y为动作种类（标签）
y = df_shuffled.iloc[:, 36 ]
# 将动作类别字段转换为one-hot-encoding
encoder = LabelEncoder()
encoded_y = encoder.fit_transform(y)
Y = np_utils.to_categorical(encoded_y)

def create_model():
    model = Sequential()
    model.add(Dense(input_dim=36, activation="relu", units=16))
    # model.add(Dropout(0.2))
    model.add(Dense(units=8, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(units=16, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(activation="softmax", units=5))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=32, verbose=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
results = cross_val_score(model, X, Y, cv=kfold)
print('Accuracy: ', np.average(results))
