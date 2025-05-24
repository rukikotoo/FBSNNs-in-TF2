# result_tf2.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# The line below is the only change, correcting the filename.
from BlackScholesBarenblatt100D_tf2 import BlackScholesBarenblatt # Directly import from the correct training script file
import os

# Disable LaTeX rendering to avoid potential errors
plt.rcParams['text.usetex'] = False

# Hyperparameters (must match the training script)
M = 100
N = 50
D = 100
layers = [D + 1] + 4 * [256] + [1]
Xi = np.array([1.0, 0.5] * int(D / 2), dtype=np.float32)[None, :]
T = 1.0

# 1. Instantiate the model
model = BlackScholesBarenblatt(Xi, T, M, N, D, layers)

# 2. Create a Checkpoint and load the parameters
checkpoint_dir = "./model_tf2"
ckpt = tf.train.Checkpoint(model=model)

# Restore the latest checkpoint
latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
if latest_ckpt:
    status = ckpt.restore(latest_ckpt)
    status.expect_partial() # 添加这一行来“压制”关于优化器的警告
    print(f"Successfully restored model from {latest_ckpt}")
else:
    print("Could not find a checkpoint to restore. Please run the training script first.")
    exit()

# 3. Generate test data
t_test, W_test = model.fetch_minibatch()

# 4. Make predictions
X_pred_tf, Y_pred_tf = model.predict(Xi, t_test, W_test)
X_pred, Y_pred = X_pred_tf.numpy(), Y_pred_tf.numpy()

# 5. Calculate the exact solution for comparison
def u_exact(t, X, T=1.0, r=0.05, sigma_max=0.4):
    return np.exp((r + sigma_max**2) * (T - t)) * np.sum(X**2, 1, keepdims=True)

Y_test = u_exact(t_test.numpy().reshape(-1, 1), 
                 X_pred.reshape(-1, D)).reshape(M, -1, 1)

# =================================================================
#               在此处添加下面的打印代码块
# =================================================================

### 7. 检查并打印初始值 (t=0) ###
# 我们检查第一条样本路径的第一个时间点的值
Y_test_0 = Y_test[0, 0, 0]
Y_pred_0 = Y_pred[0, 0, 0]
initial_error = np.abs(Y_pred_0 - Y_test_0)
relative_error = initial_error / Y_test_0 * 100

print("\n" + "="*50)
print("      检验初始点 t=0 的预测值与真实值")
print("="*50)
print(f"真实值 Y_test[0, 0, 0]:   {Y_test_0:.5f}")
print(f"预测值 Y_pred[0, 0, 0]:   {Y_pred_0:.5f}")
print("-" * 50)
print(f"绝对误差 |Y_pred - Y_test|: {initial_error:.5f}")
print(f"相对误差 (%):             {relative_error:.4f}%")
print("="*50 + "\n")

# 6. Plot the results
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

# Create a directory for the figures
os.makedirs('./figures_tf2', exist_ok=True)
plt.savefig('./figures_tf2/BSB_comparison.png')

# Plot the error
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