# ml_project_seeds
tju机器学习算法与应用大作业——基于预处理的小麦品种的分类和聚类

## 摘要
本项目基于python实现了seeds数据集的预处理与分类、聚类任务，使用了PCA、KPCA、LDA、KLDA四种算法进行数据预处理，使用SVM、逻辑回归、ANN三种方法对预处理与未预处理的数据进行了分类与评估，使用FCM方法对预处理与未预处理的数据进行了聚类与评估，完整地完成了项目的全部要求。实验过程中，对自己实现的预处理算法与sklearn的提供官方算法进行了对比；对比了预处理与否对分类与聚类精度的影响；对所有的算法均实现了可视化；基于pytorch框架使用自行搭建的MLP（多层感知机）神经网络对数据进行分类处理并总结效果。经过本次项目的实践，我对机器学习常用算法的理解与编程能力有了进一步提升，了解了预处理的重要性，也进行了不同机器学习算法应用在同一个问题上的对比，并认识到了各种算法的优劣，在日后解决科研难题的过程中，应当具体问题具体分析，选择最适合解决问题的那种算法。

## 环境配置
- 项目环境基于python3.6构建，为确保不报错，请使用python>=3.6的版本
- 建议使用conda命令进行python环境构建与依赖包的安装
```buildoutcfg
conda create -n ml_test python=3.6
activate ml_test
```
- 所需要的基本依赖包安装命令如下
```buildoutcfg
conda install numpy
conda install scikit-learn
conda install matplotlib
```
- 为运行KLDA，需要使用pip安装mlxtend扩充包
```buildoutcfg
pip install mlxtend
```
- 为运行神经网络，需要安装pytorch，使用cpu版本即可
```buildoutcfg
conda install pytorch
```

## 代码说明
- `1_1_pca_analysis.py` 自己实现的PCA预处理与sklearn实现的PCA预处理效果对比
- `1_2_kpca_analysis.py` 自己实现的KPCA预处理与sklearn实现的KPCA预处理效果对比
- `1_3_lda_analysis.py` 自己实现的LDA预处理与sklearn实现的LDA预处理效果对比
- `1_4_klda_analysis.py` sklearn未实现KLDA，这里只有自己实现的KLDA，部分源码参考[博客链接](https://blog.csdn.net/m0_37692918/article/details/102975453)
- `2_1_1_svm_raw.py` 使用SVM对原始数据直接进行分类，包括不同kernel的选取对比
- `2_1_2_svm_pca.py` 使用SVM对PCA预处理后的数据进行分类，包括不同kernel的选取对比
- `2_1_3_svm_lda.py` 使用SVM对LDA预处理后的数据进行分类，包括不同kernel的选取对比
- `2_2_1_svm_raw.py` 使用逻辑回归对原始数据直接进行分类
- `2_2_2_svm_pca.py` 使用逻辑回归对PCA预处理后的数据进行分类
- `2_2_3_svm_lda.py` 使用逻辑回归对LDA预处理后的数据进行分类
- `2_3_ann.py` 使用MLP（多层感知机）对小麦品种分类
- `3_fcm_analysis.py` 使用FCM对小麦进行聚类并评估
- `dataset.py` 读取小麦数据集的工具包