# FBSNNs_tf2.py
"""
@author: Maziar Raissi
@editor: Gemini
"""

import tensorflow as tf
import numpy as np
import time
from abc import ABC, abstractmethod

class FBSNN(tf.Module, ABC): # Forward-Backward Stochastic Neural Network
    def __init__(self, Xi, T, M, N, D, layers, name=None):
        super().__init__(name=name)
        self.Xi = tf.constant(Xi, dtype=tf.float32) # initial point
        self.T = T # terminal time
        self.M = M # number of trajectories
        self.N = N # number of time snapshots
        self.D = D # number of dimensions
        self.layers = layers # (D+1) --> 1
        
        # initialize NN
        self.weights, self.biases = self.initialize_NN(self.layers)

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1, layers[l+1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),
                           dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, t, X): # M x 1, M x D
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            u = self.neural_net(tf.concat([t, X], 1), self.weights, self.biases)
        Du = tape.gradient(u, X)
        del tape
        return u, Du

    def Dg_tf(self, X): # M x D
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X)
            g = self.g_tf(X)
        Dg = tape.gradient(g, X)
        del tape
        return Dg
        
    @tf.function
    def loss_function(self, t, W, Xi):
        loss = 0
        X_list = []
        Y_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = tf.tile(Xi, [self.M, 1])
        Y0, Z0 = self.net_u(t0, X0)

        X_list.append(X0)
        Y_list.append(Y0)

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            
            mu_val = self.mu_tf(t0, X0, Y0, Z0)
            sigma_val = self.sigma_tf(t0, X0, Y0)
            
            X1 = X0 + mu_val * (t1 - t0) + tf.squeeze(tf.matmul(sigma_val, tf.expand_dims(W1 - W0, -1)), axis=[-1])
            
            phi_val = self.phi_tf(t0, X0, Y0, Z0)
            
            Y1_tilde = Y0 + phi_val * (t1 - t0) + tf.reduce_sum(Z0 * tf.squeeze(tf.matmul(sigma_val, tf.expand_dims(W1 - W0, -1))), axis=1, keepdims=True)
            
            Y1, Z1 = self.net_u(t1, X1)
            
            loss += tf.reduce_sum(tf.square(Y1 - Y1_tilde))
            
            t0, W0, X0, Y0, Z0 = t1, W1, X1, Y1, Z1

            X_list.append(X0)
            Y_list.append(Y0)

        loss += tf.reduce_sum(tf.square(Y1 - self.g_tf(X1)))
        loss += tf.reduce_sum(tf.square(Z1 - self.Dg_tf(X1)))

        X = tf.stack(X_list, axis=1)
        Y = tf.stack(Y_list, axis=1)
        
        return loss, X, Y, Y[0,0,0]

    def fetch_minibatch(self):
        T = self.T
        M = self.M
        N = self.N
        D = self.D
        dt = T / N
        
        Dt = np.zeros((M, N + 1, 1))
        DW = np.zeros((M, N + 1, D))
        
        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))
        
        t = np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)
        
        return tf.convert_to_tensor(t, dtype=tf.float32), tf.convert_to_tensor(W, dtype=tf.float32)

    @tf.function
    def train_step(self, t_batch, W_batch):
        with tf.GradientTape() as tape:
            loss, _, _, _ = self.loss_function(t_batch, W_batch, self.Xi)
        
        trainable_variables = self.weights + self.biases
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(grads, trainable_variables))
        return loss

    def train(self, N_Iter, learning_rate):
        self.optimizer.learning_rate.assign(learning_rate)
        
        start_time = time.time()
        for it in range(N_Iter):
            t_batch, W_batch = self.fetch_minibatch()
            
            loss_value = self.train_step(t_batch, W_batch)
            
            if it % 100 == 0:
                elapsed = time.time() - start_time
                _, _, _, Y0_value = self.loss_function(t_batch, W_batch, self.Xi)
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                      (it, loss_value, Y0_value.numpy(), elapsed, self.optimizer.learning_rate.numpy()))
                start_time = time.time()

    @tf.function
    def predict(self, Xi_star, t_star, W_star):
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)
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