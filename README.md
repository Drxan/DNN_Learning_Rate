【说明】：模仿[fast.ai deep learning library中sgdr](https://github.com/fastai/fastai/blob/master/old/fastai/sgdr.py)模块进行实现。只是为了方便我在keras中使用。
# DNN_Learning_Rate
self-defined callbacks based on keras for searching and adjusting learing rate during training DNN models  

## 1 用途
### 1.1 搜索确定合适的学习率
在训练深度学习模型[或神经网络]时，其中最重要的一个超参就是学习率，它决定了每一次参数更新的幅度。合适的学习率既能缩短模型训练时间，又能使模型具
备较优的性能。论文[《Cyclical Learning Rates for Training Neural Networks》](https://arxiv.org/pdf/1506.01186.pdf)提出，首先设置一个学
习率区间[base_lr, max_lr]，在一定的迭代次数内，学习率由最初的base_lr逐步增加到max_lr，绘制出类似下面的学习率-损失的走势曲线图：
![后面补图](D:\yuwei\study\lrs_loss.png)
理想的学习率应该在损失下降的区间。
LR_Finder类用来搜索学习率，只实现了学习率的线性增长。

## 1.2 调整学习率
主要参考论文[《Cyclical Learning Rates for Training Neural Networks》](https://arxiv.org/pdf/1506.01186.pdf)提出的循环学习率策略。
CircularLR类提供两种循环学习率实现
* 固定范围的循环学习率
  即训练过程中base_lr和max_lr保持不变
* 衰减的循环学习率
  每decay_freq个周期后，对max_lr进行衰减更新，max_lr=max(max_lr,base_lr)

## 2 代码解释
* PerformanceLogger类
  该类类似keras中的History类。除了在每个epoch后记录相关指标外，还记录了每个batch后的相关性能指标、学习率等。
* LR_Updater类
  PerformanceLogger类的子类，抽象类。主要提供对学习率的更新功能。其中的抽象方法`def update_lr(self)`必须由其子类实现具体的学习率更新策略。
