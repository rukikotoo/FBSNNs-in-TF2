import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from equation import BlackScholesBarenblatt

if __name__ == "__main__":
    plt.rcParams['text.usetex'] = False

    # 超参数 (Hyperparameters)
    M = 100
    N = 50
    D = 100
    layers = [D + 1] + 4 * [256] + [1]
    Xi = np.array([1.0, 0.5] * int(D / 2), dtype=np.float32)[None, :]
    T = 1.0

    # 实例化求解器
    solver = BlackScholesBarenblatt(Xi, T, M, N, D, layers)

    # 创建检查点 (Checkpoint) 以保存模型
    checkpoint_dir = "./model_tf2"
    
    # 检查点指向求解器内部的 Keras 模型 (solver.model) 和优化器
    ckpt = tf.train.Checkpoint(model=solver.model, optimizer=solver.optimizer)
    
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)

    # 开始训练
    print("开始第一阶段训练 (学习率 1e-4)...")
    solver.train(N_Iter=1 * 10**4, learning_rate=1e-3)
    print("\n开始第二阶段训练 (学习率 1e-5)...")
    solver.train(N_Iter=1 * 10**4, learning_rate=1e-4)
    solver.train(N_Iter=1 * 10**4, learning_rate=1e-5)
    # 保存最终模型
    save_path = manager.save()
    print(f"模型已保存至: {save_path}")