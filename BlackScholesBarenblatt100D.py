"""
@author: Maziar Raissi
"""

import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
from FBSNNs import FBSNN
import matplotlib.pyplot as plt
from plotting import newfig, savefig
# 禁用LaTeX渲染避免报错
plt.rcParams['text.usetex'] = False
class BlackScholesBarenblatt(FBSNN):
    def __init__(self, Xi, T,
                       M, N, D,
                       layers):
        
        super().__init__(Xi, T,
                         M, N, D,
                         layers)
               
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return 0.05*(Y - tf.reduce_sum(X*Z, 1, keepdims = True)) # M x 1
    
    def g_tf(self, X): # M x D
        return tf.reduce_sum(X**2, 1, keepdims = True) # M x 1

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z) # M x D
        
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        return 0.4*tf.matrix_diag(X) # M x D x D
    
    ###########################################################################

if __name__ == "__main__":
    # 确保 TensorFlow 使用 GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("GPUs found:", physical_devices)
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        print("No GPU found, using CPU.")
    M = 100 # number of trajectories (batch size)
    N = 50 # number of time snapshots
    D = 100 # number of dimensions
    
    layers = [D+1] + 4*[256] + [1]

    Xi = np.array([1.0,0.5]*int(D/2))[None,:]
    T = 1.0
         
    # Training
    model = BlackScholesBarenblatt(Xi, T,
                                   M, N, D,
                                   layers)
    
    model.train(N_Iter = 1*10**4, learning_rate=1e-3)
    model.train(N_Iter = 1*10**4, learning_rate=1e-4)
    #model.train(N_Iter = 1*10**4, learning_rate=1e-5)#好像后面的训练用处不大
    #model.train(N_Iter = 1*10**4, learning_rate=1e-6)
    save_path = "./model/model.ckpt"
    saver = tf.train.Saver()
    saver.save(model.sess, save_path)  # 使用模型自身的会话（model.sess）
    print(f"模型参数已保存到：{save_path}") 