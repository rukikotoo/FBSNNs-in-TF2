# BlackScholesBarenblatt100D_tf2.py (Corrected)
"""
@author: Maziar Raissi
@editor: Gemini
"""
import tensorflow as tf
import numpy as np
import os
from FBSNNs_tf2 import FBSNN  # <--- 修正了导入，现在导入基类 FBSNN

#
# 类的定义现在被添加回了这个文件
#
class BlackScholesBarenblatt(FBSNN):
    """
    Inherits from the abstract base class FBSNN and implements the specific
    PDE functions (phi, g, mu, sigma).
    """
    def __init__(self, Xi, T, M, N, D, layers):
        # Pass all arguments to the parent class constructor
        super().__init__(Xi, T, M, N, D, layers)

    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        # Note: keepdims=True is the TF2 equivalent of keep_dims=True
        return 0.05 * (Y - tf.reduce_sum(X * Z, axis=1, keepdims=True)) # M x 1

    def g_tf(self, X): # M x D
        return tf.reduce_sum(tf.square(X), axis=1, keepdims=True) # M x 1

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z) # M x D

    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        # TF1's tf.matrix_diag is tf.linalg.diag in TF2
        return 0.4 * tf.linalg.diag(X) # M x D x D

#
# 主训练脚本部分保持不变
#
if __name__ == "__main__":
    # 确保不使用 Matplotlib 的 LaTeX 渲染以避免潜在错误
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = False

    # Hyperparameters
    M = 100  # number of trajectories (batch size)
    N = 50   # number of time snapshots
    D = 100  # number of dimensions
    layers = [D + 1] + 4 * [256] + [1]
    Xi = np.array([1.0, 0.5] * int(D / 2), dtype=np.float32)[None, :]
    T = 1.0

    # Initialize the model
    # 现在 'BlackScholesBarenblatt' 类在上面已经定义，所以这行代码可以正常工作
    model = BlackScholesBarenblatt(Xi, T, M, N, D, layers)

    # Create a Checkpoint for saving the model
    checkpoint_dir = "./model_tf2"
    ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)

    # Start training
    print("Starting training with learning rate 1e-3...")
    model.train(N_Iter=1 * 10**4, learning_rate=1e-3)

    print("\nStarting training with learning rate 1e-4...")
    model.train(N_Iter=1 * 10**4, learning_rate=1e-4)

    # Save the final model
    save_path = manager.save()
    print(f"Model saved to {save_path}")