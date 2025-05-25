# equation.py 
import tensorflow as tf
from FBSNNs_tf2 import FBSNN_Solver 

class BlackScholesBarenblatt(FBSNN_Solver):
    """
    这个类现在继承自通用的求解器 FBSNN_Solver。
    它实现了特定于 Black-Scholes 方程的函数。
    """
    def __init__(self, Xi, T, M, N, D, layers):
        # 调用新的父类 FBSNN_Solver 的构造函数
        super().__init__(Xi, T, M, N, D, layers)
        # 方程相关的常数
        self.r = 0.05
        self.sigma_val = 0.4

    def phi_tf(self, t, X, Y, Z):
        return self.r * (Y - tf.reduce_sum(X * Z, axis=1, keepdims=True))

    def g_tf(self, X):
        return tf.reduce_sum(tf.square(X), axis=1, keepdims=True)

    def mu_tf(self, t, X, Y, Z):
        return super().mu_tf(t, X, Y, Z)

    def sigma_tf(self, t, X, Y):
        # 扩散项，与您的代码一致
        # 返回一个对角矩阵 (M, D, D)
        return self.sigma_val * tf.linalg.diag(X)
class HamiltonJacobiBellman(FBSNN_Solver):
    
    def __init__(self, Xi, T,
                 M, N, D,
                 layers):
        
        super().__init__(Xi, T,
                         M, N, D,
                         layers)
    
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
      
        return tf.reduce_sum(tf.square(Z), axis=1, keepdims=True) # M x 1
    
    def g_tf(self, X): # M x D
      
        return tf.math.log(0.5 + 0.5*tf.reduce_sum(tf.square(X), axis=1, keepdims = True)) # M x 1

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
      
        return super().mu_tf(t, X, Y, Z) # M x D
    
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
       
        return tf.math.sqrt(2.0)*super().sigma_tf(t, X, Y) # M x D x D
class AllenCahn(FBSNN_Solver):
    def __init__(self, Xi, T,
                       M, N, D,
                       layers):
        
        super().__init__(Xi, T,
                         M, N, D,
                         layers)
    
    def phi_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return - Y + Y**3 # M x 1
    
    def g_tf(self, X):
        return 1.0/(2.0 + 0.4*tf.reduce_sum(X**2, 1, keepdims = True))

    def mu_tf(self, t, X, Y, Z): # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z) # M x D
    
    def sigma_tf(self, t, X, Y): # M x 1, M x D, M x 1
        return super().sigma_tf(t, X, Y) # M x D x D