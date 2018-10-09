# DNN_Learning_Rate
self-defined callbacks based on keras for searching and adjusting learing rate during training DNN models
## 1 用途
### 1.1 搜索确定合适的学习率
在训练深度学习模型[或神经网络]时，其中最重要的一个超参就是学习率，它决定了每一次参数更新的幅度。合适的学习率既能缩短模型训练时间，又能使模型具
备较优的性能。论文[《Cyclical Learning Rates for Training Neural Networks》](https://arxiv.org/pdf/1506.01186.pdf)提出，首先设置一个学
习率区间[base_lr, max_lr]，在一定的迭代次数内，学习率由最初的base_lr逐步增加到max_lr，绘制出类似下面的学习率-损失的走势曲线图：
![OMG][lr_loss]
模仿fast.ai deep learning library进行实现。
