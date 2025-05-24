# FBSNNs-in-TF2
在用tf2重写项目之前，我先基于tf1跑通项目，理解内在逻辑。tf2是兼容tf1的，只需要改为
```python
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
```

## 依赖环境
tensorflow 2.9.0 

Python	3.10.x	

CUDA	11.2	

cuDNN	8.1.0	

numpy	1.26.4	

这一点原文没有说明，我尝试了好久才配好环境。值得一提的是，在虚拟环境中装好CUDA和cuDNN之后需要在系统变量中的path设置虚拟环境的地址，不然还是会调用全局的cuda，导致版本不匹配。

## 项目逻辑
FBSNNS.py 脚本为算法提供了一个框架，其中预留了四个核心的抽象方法：phi_tf、g_tf、mu_tf 和 sigma_tf,对应随机微分方程中的函数。使用者可以根据具体的任务带入这些函数来实现求解。
```math
\begin{aligned}
    dX_t &= \mu(t, X_t, Y_t, Z_t)dt + \sigma(t, X_t, Y_t)dW_t, && t \in [0, T], \\
    X_0 &= \xi, \\
    dY_t &= \varphi(t, X_t, Y_t, Z_t)dt + Z_t'\sigma(t, X_t, Y_t)dW_t, && t \in [0, T], \\
    Y_T &= g(X_T),
\end{aligned}
```

原文中展示了 Black-Scholes-Barenblatt (BSB) 方程和 HJB 方程两个案例。为了聚焦核心逻辑并节约复现时间，本项目仅实现了 BSB 方程。

原作者的代码在训练结束后直接生成了可视化结果，这种方式不利于模型的复用和评估。我对此作出了改进：

分离训练与测试：将训练 (train) 和测试 (test) 过程解耦为独立的模块。

模型持久化：训练好的模型参数会被保存到 /model 文件夹中。这样，你可以随时加载已训练的模型进行测试、评估或继续训练。

## 训练复现

在我的设备上，CPU 训练一轮（iteration）耗时约 1.x 秒，而 GPU 耗时约 0.x 秒，性能提升并不显著。
考虑到时间成本和边际效益，我调整了训练策略，原文使用学习率衰减训练了 10 万轮。我只训练了 2 万轮，训练中仅手动调整了一次学习率。实践证明，在训练后期，进一步增加轮数对 loss 的降低贡献非常有限。

可以在 /figures 文件夹中找到复现的结果图：

BSB_524_50.pdf & BSB_524_errors.pdf: 我的复现结果

BSB_Apr18_50.pdf & BSB_Apr18_50_errors.pdf: 原文的结果

从结果图可以看出，尽管我的训练轮数远少于原文，但最终的预测误差非常接近，甚至略优。在误差图中，红色虚线代表 “均值 + 两个标准差”，可以将其理解为误差的置信区间上界。





