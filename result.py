
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from plotting import newfig, savefig
from FBSNNs import FBSNN
import os

# 禁用LaTeX渲染避免报错
plt.rcParams['text.usetex'] = False

# 定义相同的模型类
class BlackScholesBarenblatt(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers):
        super().__init__(Xi, T, M, N, D, layers)
    
    def phi_tf(self, t, X, Y, Z): 
        return 0.05*(Y - tf.reduce_sum(X*Z, 1, keepdims = True))
    
    def g_tf(self, X): 
        return tf.reduce_sum(X**2, 1, keepdims = True)

    def mu_tf(self, t, X, Y, Z): 
        return super().mu_tf(t, X, Y, Z)
        
    def sigma_tf(self, t, X, Y): 
        return 0.4*tf.matrix_diag(X)

# 超参数（需与训练时一致）
M = 100
N = 50
D = 100
layers = [D+1] + 4*[256] + [1]
Xi = np.array([1.0,0.5]*int(D/2))[None,:]
T = 1.0

# 1. 重建计算图
tf.reset_default_graph()  # 清除之前的计算图
model = BlackScholesBarenblatt(Xi, T, M, N, D, layers)

# 2. 创建Saver并加载参数
saver = tf.train.Saver()
sess = model.sess 
# 加载检查点
saver.restore(sess, "./model/model.ckpt")
print("成功加载模型参数！")
    
# 3. 生成测试数据
t_test, W_test = model.fetch_minibatch()
    
# 4. 进行预测
X_pred, Y_pred = model.predict(Xi, t_test, W_test)
    
# 5. 计算真实值
def u_exact(t, X):
    r = 0.05
    sigma_max = 0.4
    return np.exp((r + sigma_max**2)*(T - t))*np.sum(X**2, 1, keepdims = True)
    
Y_test = np.reshape(u_exact(np.reshape(t_test[0:M,:,:],[-1,1]), 
                              np.reshape(X_pred[0:M,:,:],[-1,D])),[M,-1,1])
print("权重示例（第一层）：", sess.run(model.weights[0][0, :5]))  # 打印第一层前5个权重
t_test, W_test = model.fetch_minibatch()
print("t_test 范围：", np.min(t_test), np.max(t_test))  # 应为 [0.0, T]
print("W_test 均值：", np.mean(W_test))                 # 应接近 0，但方差为 dt
# 打印预测值，确认不再是0
print("Y_pred 示例值:", Y_pred[0, 0:5, 0]) 
# 6. 绘制结果图
samples = 5
plt.figure(figsize=(10,6))
plt.plot(t_test[0:1,:,0].T,Y_pred[0:1,:,0].T,'b',label='Learned $u(t,X_t)$')
plt.plot(t_test[0:1,:,0].T,Y_test[0:1,:,0].T,'r--',label='Exact $u(t,X_t)$')
plt.plot(t_test[0:1,-1,0],Y_test[0:1,-1,0],'ko',label='$Y_T = u(T,X_T)$')
    
for i in range(1, samples):
    plt.plot(t_test[i,:,0].T, Y_pred[i,:,0].T, 'b')
    plt.plot(t_test[i,:,0].T, Y_test[i,:,0].T, 'r--')
    plt.plot(t_test[i,-1,0], Y_test[i,-1,0], 'ko')

plt.plot([0], Y_test[0,0,0], 'ks', label='$Y_0 = u(0,X_0)$')
plt.xlabel('$t$')
plt.ylabel('$Y_t = u(t,X_t)$')
plt.title('100-dimensional Black-Scholes-Barenblatt')
plt.legend()
    
# 创建保存目录
os.makedirs('./figures', exist_ok=True)
savefig('./figures/BSB_524_50', crop=False)
    
# 绘制误差图
errors = np.sqrt((Y_test-Y_pred)**2/Y_test**2)
mean_errors = np.mean(errors,0)
std_errors = np.std(errors,0)
    
plt.figure(figsize=(10,6))
plt.plot(t_test[0,:,0], mean_errors, 'b', label='mean')
plt.plot(t_test[0,:,0], mean_errors+2*std_errors, 'r--', 
             label='mean + two standard deviations')
plt.xlabel('$t$')
plt.ylabel('relative error')
plt.title('100-dimensional Black-Scholes-Barenblatt')
plt.legend()
    
savefig('./figures/BSB_524_errors', crop=False)
plt.show()
sess.close()
   
    