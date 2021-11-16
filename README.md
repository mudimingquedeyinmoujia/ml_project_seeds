# ml_project_seeds
tju机器学习算法与应用大作业——基于预处理的小麦品种的分类和聚类

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