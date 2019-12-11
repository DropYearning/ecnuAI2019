# 华东师范大学-研究生课程-人工智能基础 -2019
基于机器学习的姿态估计和动作识别

## 项目目录
程序目录如下：

├── action_recognize:动作识别主目录
│   ├── model:存放预训练好的动作识别模型
│   │   ├── cnn_model_architecture.json
│   │   ├── cnn_model_weights.h5
│   │   ├── decision_tree_model.m
│   │   ├── dnn_model.h5
│   │   ├── knn_model.m
│   │   ├── naivebayes_model.m
│   │   ├── rfc_model.m
│   │   └── svm_model.m
│   └── train: 每个文件夹下存放对应的图片数据集
│       ├── CNN
│       ├── DNN
│       ├── DecisionTree
│       ├── KNN
│       ├── NaiveBayes
│       ├── RFC
│       └── SVM
├── conda_environment.yaml: conda依赖环境
├── pip_packages.txt: pip依赖环境
├── csv: 存放一些姿态估计得到的csv文件
├── get_keypoints_position.py: Python程序，从图片中获取关节位置
├── images: 存放动作识别的输入图片
├── input: 存放姿态估计的输入图片（共计5类动作，图片省略）
│   ├── bowling
│   ├── flap
│   ├── squat
│   ├── stand
│   ├── test
│   └── wave
├── load_cnn.py: Python程序，载入CNN模型
├── models:存放OpenPose的预训练模型
├── normalization.py: Python程序，归一化处理
├── openpose_run.py: Python程序，对一张图片进行姿态估计
├── openpose_run_directory.py: 对一个目录下的图片进行姿态估计
├── predict.py: Python程序，对图片进行动作识别，窗口展示
├── tf_pose:tf-open-pose主目录
└── 截图:存放了一些项目截图

参考项目:
- [CMU-Perceptual-Computing-Lab/openpose: OpenPose: Real-time multi-person keypoint detection library for body, face, hands, and foot estimation](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 
[ildoonet/tf-pose-estimation: Deep Pose Estimation implemented using Tensorflow with Custom Architectures for fast inference.](https://github.com/ildoonet/tf-pose-estimation)
- [LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose: A skeleton-based real-time online action recognition project, classifying and recognizing base on framewise joints, which can be used for safety surveilence.](https://github.com/LZQthePlane/Online-Realtime-Action-Recognition-based-on-OpenPose)
- [基于骨架图的实时视频行为识别系统 - 简书](https://www.jianshu.com/p/733fa86b0d8b)
- [felixchenfy/Realtime-Action-Recognition: Multi-person real-time recognition (classification) of 9 actions based on human skeleton from OpenPose and a 0.5-second window.](https://github.com/felixchenfy/Realtime-Action-Recognition)

