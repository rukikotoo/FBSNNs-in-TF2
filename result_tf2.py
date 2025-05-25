import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from equation import BlackScholesBarenblatt 

# 关闭 LaTeX 渲染以避免潜在错误
plt.rcParams['text.usetex'] = False

# 超参数 (必须与训练脚本完全一致)
M = 100
N = 50
D = 100
layers = [D + 1] + 4 * [256] + [1]
Xi = np.array([1.0, 0.5] * int(D / 2), dtype=np.float32)[None, :]
T = 1.0

# 1. 实例化求解器对象
solver = BlackScholesBarenblatt(Xi, T, M, N, D, layers)

# 2. 创建检查点并加载已训练的参数
# <--- 修改：指向新训练脚本保存的目录
checkpoint_dir = "./model_tf2" 

# <--- 关键修改：Checkpoint 对象必须与保存时的结构匹配。
# 我们保存的是 solver.model，所以这里也必须指向 solver.model。
ckpt = tf.train.Checkpoint(model=solver.model)

# 恢复最新的检查点
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_ckpt:
    # 仅恢复模型的权重，优化器的状态在推理时不需要
    status = ckpt.restore(latest_ckpt)
    # 推荐使用 expect_partial()，因为它明确告诉 TensorFlow 我们只期望恢复模型权重，
    # 即使检查点文件中还包含优化器状态也不会发出警告。
    status.expect_partial() 
    print(f"成功从 {latest_ckpt} 恢复模型权重")
else:
    print("找不到可以恢复的检查点。请先运行训练脚本。")
    exit()

# 3. 生成测试数据
# <--- 修改：从 solver 对象调用函数
t_test, W_test = solver.fetch_minibatch()

# 4. 进行预测
# <--- 修改：从 solver 对象调用函数
X_pred_tf, Y_pred_tf = solver.predict(Xi, t_test, W_test)
X_pred, Y_pred = X_pred_tf.numpy(), Y_pred_tf.numpy()

# 5. 计算用于比较的精确解 
def u_exact(t, X, T=1.0, r=0.05, sigma_max=0.4):
    return np.exp((r + sigma_max**2) * (T - t)) * np.sum(X**2, 1, keepdims=True)

# 注意reshape的顺序，先-1再D，确保每行是一个D维向量
Y_test = u_exact(t_test.numpy().reshape(-1, 1), 
                 X_pred.reshape(-1, D)).reshape(M, -1, 1)

# 6. 检查并打印初始值 (t=0) - 您的代码块，原样保留
### 检查并打印初始值 (t=0) ###
# 我们检查第一条样本路径的第一个时间点的值
Y_test_0 = Y_test[0, 0, 0]
Y_pred_0 = Y_pred[0, 0, 0]
initial_error = np.abs(Y_pred_0 - Y_test_0)
relative_error = initial_error / Y_test_0 * 100

print("\n" + "="*50)
print("       检验初始点 t=0 的预测值与真实值")
print("="*50)
print(f"真实值 Y_test[0, 0, 0]:   {Y_test_0:.5f}")
print(f"预测值 Y_pred[0, 0, 0]:   {Y_pred_0:.5f}")
print("-" * 50)
print(f"绝对误差 |Y_pred - Y_test|: {initial_error:.5f}")
print(f"相对误差 (%):             {relative_error:.4f}%")
print("="*50 + "\n")

# 7. 绘制结果图 (这部分基于 numpy 数组，无需修改)
samples = 5
plt.figure(figsize=(10, 6))
t_plot = t_test.numpy()

plt.plot(t_plot[0, :, 0].T, Y_pred[0, :, 0].T, 'b', label='Learned $u(t,X_t)$')
plt.plot(t_plot[0, :, 0].T, Y_test[0, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
plt.plot(t_plot[0, -1, 0], Y_test[0, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')

for i in range(1, samples):
    plt.plot(t_plot[i, :, 0].T, Y_pred[i, :, 0].T, 'b')
    plt.plot(t_plot[i, :, 0].T, Y_test[i, :, 0].T, 'r--')
    plt.plot(t_plot[i, -1, 0], Y_test[i, -1, 0], 'ko')

plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')
plt.xlabel('$t$')
plt.ylabel('$Y_t = u(t,X_t)$')
plt.title('100-dimensional Black-Scholes-Barenblatt (TF2)')
plt.legend()

# <--- 修改：为图片创建新目录
os.makedirs('./figures_tf2', exist_ok=True)
plt.savefig('./figures_tf2/BSB_comparison.png')

# 绘制误差图
errors = np.sqrt((Y_test - Y_pred)**2 / Y_test**2)
mean_errors = np.mean(errors, 0)
std_errors = np.std(errors, 0)

plt.figure(figsize=(10, 6))
plt.plot(t_plot[0, :, 0], mean_errors, 'b', label='mean')
plt.plot(t_plot[0, :, 0], mean_errors + 2 * std_errors, 'r--',
         label='mean + two standard deviations')
plt.xlabel('$t$')
plt.ylabel('relative error')
plt.title('100-dimensional Black-Scholes-Barenblatt (TF2)')
plt.legend()

plt.savefig('./figures_tf2/BSB_errors.png')
plt.show()