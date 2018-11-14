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

### 1.2 调整学习率
主要参考论文[《Cyclical Learning Rates for Training Neural Networks》](https://arxiv.org/pdf/1506.01186.pdf)提出的循环学习率策略。
CircularLR类提供两种循环学习率实现
* 固定范围的循环学习率
  即训练过程中base_lr和max_lr保持不变
* 衰减的循环学习率
  每decay_freq个周期后，对max_lr进行衰减更新，max_lr=max(max_lr,base_lr)
* 其他不同的学习率更新策略都可以继承LR_Updater类进行具体实现。

## 2 代码解释
* PerformanceLogger类
  该类类似keras中的History类。除了在每个epoch后记录相关指标外，还记录了每个batch后的相关性能指标、学习率等。该类及其子类对象在记录这些指标后可以把存下来，以便分析模型性能，为模型调优做参考。
* LR_Updater类
  PerformanceLogger类的子类，抽象类。主要提供对学习率的更新功能。其中的抽象方法`def update_lr(self)`必须由其子类实现具体的学习率更新策略。

## 3 用法
### 3.1 学习率搜索 
lr_finder = LR_Finder(base_lr=1e-9, lr_multiplier=1.06)
cbks = [lr_finder]
train_hist = model.fit(train_X, train_y, batch_size=batch_size, epochs=1000, verbose=2, validation_data=(val_X, val_y), callbacks=cbks)  
(1) 利用自带的绘图函数  
  lr_finder.plot_loss() 

（2）自定义绘图函数  
`def plot_lrs(hist, low=0, up=-1, marker='', x_log=True, moving_avg=False, alpha=0.9):
    print('iter_num:',len(hist['lrs']))
    plt.rcParams['figure.figsize']=(12,6)
    x = hist['lrs']
    y = hist['loss']
    if moving_avg:
        ma = y[0]
        for i in range(len(y)):
            ma = ma*alpha + y[i]*(1-alpha)
            y[i] = ma
    plt.grid(axis='x')
    plt.plot(x[low:up],y[low:up],ls='-',marker=marker)
    # plt.xticks(rotation=30)
    if x_log:
        plt.xscale('log')
    plt.xlabel('lrs')
    plt.ylabel('loss')
    plt.title('lr-loss')
    return plt.xticks()
    
 hist = lr_finder.batch_history
 plot_lrs（hist, moving_avg=True）` 
 
 ### 3.2 循环学习率设置 
 step_size 表示循环的半周期长度（mini_bath的数量)  
`clr = CircularLR(step_size=step_size, base_lr=base_lr, max_lr=max_lr, decay=0.6, decay_type='exp')
cbks = [clr] 
train_hist = model.fit(train_X, train_y, batch_size=batch_size, epochs=1000, verbose=2, validation_data=(val_X, val_y), callbacks=cbks)`
