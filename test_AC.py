import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from equation import AllenCahn


# 关闭 LaTeX 渲染以避免潜在错误
plt.rcParams['text.usetex'] = False

# 超参数 (必须与训练脚本完全一致)
M = 100
N = 50
D = 20 
layers = [D + 1] + 4 * [256] + [1]
Xi = np.zeros([1, D], dtype=np.float32) # Allen-Cahn 通常从 0 开始
T = 0.3 # Allen-Cahn 的时间通常较短

# 1. 实例化求解器对象
solver = AllenCahn(Xi, T, M, N, D, layers)

# 2. 创建检查点并加载已训练的参数
checkpoint_dir = "./model_tf2_AC" 
ckpt = tf.train.Checkpoint(model=solver.model)

# 恢复最新的检查点
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_ckpt:
    status = ckpt.restore(latest_ckpt)
    status.expect_partial() 
    print(f"成功从 {latest_ckpt} 恢复模型权重")
else:
    print("找不到可以恢复的检查点。请先运行训练脚本。")
    exit()

# 3. 生成数据并进行预测
# <--- 修改 #1：使用正确的 'solver' 对象调用函数
t_test_tf, W_test_tf = solver.fetch_minibatch()
X_pred_tf, Y_pred_tf = solver.predict(Xi, t_test_tf, W_test_tf)

# 4. <--- 修改 #2：将 TensorFlow 张量转换为 NumPy 数组 --->
# 这是为了后续的绘图和基于 NumPy 的计算
t_test = t_test_tf.numpy()
X_pred = X_pred_tf.numpy()
Y_pred = Y_pred_tf.numpy()

# 5. 计算终止时刻的值
# 现在 X_pred 是 NumPy 数组，这行代码可以正常工作
Y_test_terminal = 1.0/(2.0 + 0.4*np.sum(X_pred[:,-1,:]**2, axis=1, keepdims=True))

# 6. 绘图
# 现在 t_test 和 Y_pred 都是 NumPy 数组，绘图可以正常进行
samples = 5
plt.figure(figsize=(10, 6)) # 增加图像尺寸以便观察
plt.plot(t_test[0,:,0].T, Y_pred[0,:,0].T, 'b', label='Learned $u(t,X_t)$')
# 确保切片范围不超过样本数
plot_samples = min(samples, M)
plt.plot(t_test[1:plot_samples,:,0].T, Y_pred[1:plot_samples,:,0].T, 'b')
plt.plot(t_test[0:plot_samples,-1,0], Y_test_terminal[0:plot_samples,0], 'ks', label='$Y_T = u(T,X_T)$')

# 对于Allen-Cahn问题，通常没有简单的精确解，但有时会与一个已知点比较
# 您的代码中有一个参考点，我们保留它
plt.plot([0],[0.30879],'ro', label='Reference $Y_0$') # 改为红色圆点以突出
plt.xlabel('$t$')
plt.ylabel('$Y_t = u(t,X_t)$')
# <--- 修改 #3 (推荐)：让标题动态匹配维度D
plt.title(f'{D}-dimensional Allen-Cahn')
plt.legend()

# 7. 保存图像
os.makedirs('./figures_AC/', exist_ok=True)
plt.savefig('./figures_AC/AC_comparison.png')
plt.show()