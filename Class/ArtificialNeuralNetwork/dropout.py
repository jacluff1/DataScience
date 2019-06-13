import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

D = 2
K = 3
N = int(K*1e3)

X0 = np.random.randn((N//K),D) + np.array([2,2])
X1 = np.random.randn((N//K),D) + np.array([0,-2])
X2 = np.random.randn((N//K),D) + np.array([-2,2])
X = np.vstack((X0,X1,X2))

y = n.array([0]*(N//K) + [1]*(N//K) + [2]*(N//K))

def one_hot_encode(y):
    N = len(y)
    K = len(set(y))
    Y = np.zeros((N,K))
    Y[i,y[i]] = 1
    return Y

def ReLU(H):
    return H * (H > 0)

def softmax(H):
    eH = np.exp(H)
    return eH / eH.sum(axis=1, keepdims=True)

def feed_forward_train(X,W,b,p_keep):

    Z = {}
    N,D = X.shape
    L = len(W)

    mask = 1*(p_keep[1] >= np.random.rand(N,D))
    Z[1] = ReLU( np.matmul(X*X[mask],W[1]) + b[1] )

    for l in range(1,L):
        mask = 1*(p_keep[l] >= np.random.rand(N,W.shape[0]))
        Z[l] = ReLU( np.matmul(Z[l-1]*Z[l-1][mask],W[l]) + b[l] )

    mask = 1*(p_keep[L] <= np.random.rand(N,K))
    Z[L] = softmax( np.matmul(Z[L]*Z[L][mask],W[L]) + b[L] )

    return Z

def feed_forward_predict(X,W,b,p_keep):

    Z = {}
    N = X.shape[0]
    L = len(W)

    Z[1] = ReLU( np.matmul((1/p_keep[1])*X,W[1] ) + b[1] )
    for l in range(1,L):
        Z[l] = ReLU( np.matmul((1/p_keep[l])*Z[l-1],W[l] ) + b[l] )
    Z[L] = softmax( np.matmul((1/p_keep[L])*Z[L],W[L] ) + b[L] )

    return Z[L]



plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, alpha=0.001)
