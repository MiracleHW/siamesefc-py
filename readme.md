## siamese-fc for python using tensorflow

Tracking的算法siamese-fc使用的是matlab和matconvnet

现在用python进行重写，使用tensorflow来构建网络，直接套用作者训练好的参数

原作者论文：https://arxiv.org/pdf/1606.09549v2.pdf

原作者的代码：https://github.com/bertinetto/siamese-fc

所需环境：opencv2.4，tensorflow(>0.10)

直接运行tracking类里的tracker方法

如果测试自定义视频，需要重写tracking类里的ReadSequences方法
