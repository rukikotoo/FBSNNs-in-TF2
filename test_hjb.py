import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from equation import HamiltonJacobiBellman

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
solver = HamiltonJacobiBellman(Xi, T, M, N, D, layers)

# 2. 创建检查点并加载已训练的参数
checkpoint_dir = "./model_tf2_hjb" 
ckpt = tf.train.Checkpoint(model=solver.model)
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_ckpt:
    status = ckpt.restore(latest_ckpt)
    status.expect_partial() 
    print(f"成功从 {latest_ckpt} 恢复模型权重")
else:
    print("找不到可以恢复的检查点。请先运行训练脚本。")
    exit()

# 3. 生成数据并进行预测
t_test_tf, W_test_tf = solver.fetch_minibatch()
X_pred_tf, Y_pred_tf = solver.predict(Xi, t_test_tf, W_test_tf)

# 4. <--- 关键修改：将 TensorFlow 张量转换为 NumPy 数组 --->
# 这是为了后续的绘图和基于 NumPy 的计算
t_test = t_test_tf.numpy()
X_pred = X_pred_tf.numpy()
Y_pred = Y_pred_tf.numpy()

# 5. 计算精确解
def g(X): # MC x NC x D
    return np.log(0.5 + 0.5*np.sum(X**2, axis=2, keepdims=True)) # MC x N x 1
    
def u_exact(t, X): # NC x 1, NC x D
    MC = 10**5
    NC = t.shape[0]
    W = np.random.normal(size=(MC,NC,D)) # MC x NC x D
    return -np.log(np.mean(np.exp(-g(X + np.sqrt(2.0*np.abs(T-t))*W)),axis=0))

Y_test = u_exact(t_test[0,:,:], X_pred[0,:,:])
Y_test_terminal = np.log(0.5 + 0.5*np.sum(X_pred[:,-1,:]**2, axis=1, keepdims=True))

# 6. 绘图
plt.figure()
plt.plot(t_test[0,:,0].T, Y_pred[0,:,0].T, 'b', label='Learned $u(t,X_t)$')
plt.plot(t_test[0,:,0].T, Y_test[:,0].T, 'r--', label='Exact $u(t,X_t)$')
plt.plot(t_test[0,-1,0], Y_test_terminal[0,0], 'ks', label='$Y_T = u(T,X_T)$')
plt.plot([0],Y_test[0,0],'ko',label='$Y_0 = u(0,X_0)$')
plt.xlabel('$t$')
plt.ylabel('$Y_t = u(t,X_t)$')
plt.title('100-dimensional Hamilton-Jacobi-Bellman')
plt.legend()
os.makedirs('./figures_HJB/', exist_ok=True)
plt.savefig('./figures_HJB/HJB_comparison.png')

# 7. 绘制误差图
errors = np.sqrt((Y_test - Y_pred[0,:,:])**2 / Y_test**2)
plt.figure()
plt.plot(t_test[0,:,0], errors, 'b')
plt.xlabel('$t$')
plt.ylabel('relative error')
plt.title('100-dimensional Hamilton-Jacobi-Bellman')
plt.savefig('./figures_HJB/HJB_errors.png')
plt.show()