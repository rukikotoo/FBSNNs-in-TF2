# FBSNNs_tf2.py
"""
@author: Maziar Raissi

"""
import tensorflow as tf
import numpy as np
import time
from abc import ABC, abstractmethod

# ==============================================================================
# 1. 神经网络模型类
#    - 负责定义网络架构。
#    - 继承自 tf.keras.Model 以实现更好的集成。
# ==============================================================================
class FBSNN_Model(tf.keras.Model):
    """
    定义前馈神经网络的架构。
    结构: (D+1) --> 隐藏层 --> 1
    """
    def __init__(self, layers, name=None):
        super().__init__(name=name)
        self.hidden_layers = []
        # 使用 Keras 标准的 Glorot (Xavier) 初始化器
        initializer = tf.keras.initializers.GlorotUniform()

        # 创建所有隐藏层，并使用 sin 作为激活函数
        for i in range(len(layers) - 2):
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    layers[i+1],
                    activation=tf.sin,
                    use_bias=True,
                    kernel_initializer=initializer
                )
            )

        # 创建线性的输出层
        self.output_layer = tf.keras.layers.Dense(
            layers[-1],
            activation=None,
            use_bias=True,
            kernel_initializer=initializer
        )

    def call(self, x):
        """网络的前向传播。"""
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
        return self.output_layer(h)
#当你通过继承 tf.keras.Model 或 tf.keras.layers.Layer 来创建自定义模型/层时，你必须将你的运算逻辑放在名为 call 的方法里。
# ==============================================================================
# 2. FBSNN 求解器类
#    - 包含求解 FBSDE 的核心逻辑。
#    - 管理模拟、损失计算和训练过程。
#    - 使用一个神经网络模型的实例来辅助计算。
# ==============================================================================
class FBSNN_Solver(ABC):
    """前向-后向随机神经网络求解器。"""
    def __init__(self, Xi, T, M, N, D, layers):
        self.Xi = tf.constant(Xi, dtype=tf.float32)
        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.model = FBSNN_Model(layers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def net_u(self, t, X):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            input_data = tf.concat([t, X], axis=1)
            u = self.model(input_data)
        Du = tape.gradient(u, X)
        del tape
        return u, Du

    def Dg_tf(self, X):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            g = self.g_tf(X)
        Dg = tape.gradient(g, X)
        del tape
        return Dg
    
    @tf.function
    def loss_function(self, t, W):
        loss = tf.constant(0.0, dtype=tf.float32)
        X_list = []
        Y_list = []

        t0 = t[:, 0, :]
        X0 = tf.tile(self.Xi, [self.M, 1])
        Y0, Z0 = self.net_u(t0, X0)

        X_list.append(X0)
        Y_list.append(Y0)

        dt = self.T / self.N
        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            dW = tf.expand_dims(W1 - W[:, n, :], -1)
            
            mu_val = self.mu_tf(t0, X0, Y0, Z0)
            sigma_val = self.sigma_tf(t0, X0, Y0)
            X1 = X0 + mu_val * dt + tf.squeeze(tf.matmul(sigma_val, dW), axis=[-1])

            phi_val = self.phi_tf(t0, X0, Y0, Z0)
            
            Y1_tilde = Y0 + phi_val * dt + tf.reduce_sum(Z0 * tf.squeeze(tf.matmul(sigma_val, dW)), axis=1, keepdims=True)
            
            Y1, Z1 = self.net_u(t1, X1)
            loss += tf.reduce_sum(tf.square(Y1 - Y1_tilde))
            
            t0, X0, Y0, Z0 = t1, X1, Y1, Z1
            X_list.append(X0)
            Y_list.append(Y0)

        loss += tf.reduce_sum(tf.square(Y1 - self.g_tf(X1)))
        loss += tf.reduce_sum(tf.square(Z1 - self.Dg_tf(X1)))
        
        X = tf.stack(X_list, axis=1)
        Y = tf.stack(Y_list, axis=1)
        
        return loss, X, Y, Y[0, 0, 0]

    def fetch_minibatch(self):
        dt = self.T / self.N
        dW = np.sqrt(dt) * np.random.normal(size=(self.M, self.N, self.D))
        dW_padded = np.zeros((self.M, self.N + 1, self.D))
        dW_padded[:, 1:, :] = dW
        W = np.cumsum(dW_padded, axis=1)
        t = np.linspace(0, self.T, self.N + 1).reshape(1, -1, 1)
        t_batch = np.tile(t, (self.M, 1, 1))
        return tf.convert_to_tensor(t_batch, dtype=tf.float32), tf.convert_to_tensor(W, dtype=tf.float32)

    @tf.function
    def train_step(self, t_batch, W_batch):
        with tf.GradientTape() as tape:
            loss, _, _, _ = self.loss_function(t_batch, W_batch)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, N_Iter, learning_rate):
        self.optimizer.learning_rate.assign(learning_rate)
        start_time = time.time()
        for it in range(N_Iter):
            t_batch, W_batch = self.fetch_minibatch()
            loss_value = self.train_step(t_batch, W_batch)
            if it % 100 == 0:
                elapsed = time.time() - start_time
                _, _, _, Y0_value = self.loss_function(t_batch, W_batch)
                print(f'It: {it:5d}, Loss: {loss_value:.3e}, Y0: {Y0_value.numpy():.3f}, Time: {elapsed:.2f}, LR: {self.optimizer.learning_rate.numpy():.3e}')
                start_time = time.time()

    def predict(self, Xi_star, t_star, W_star):
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star)
        return X_star, Y_star

    @abstractmethod
    def phi_tf(self, t, X, Y, Z): pass

    @abstractmethod
    def g_tf(self, X): pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        return tf.zeros([self.M, self.D], dtype=tf.float32)

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        return tf.linalg.diag(tf.ones([self.M, self.D], dtype=tf.float32))