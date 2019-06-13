import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from DataScience.ANN import ArtificalNeuralNet
import DataScience.ActivationFunctions as AF

D = 2
K = 3
N = int(K*1.5e4)

X0 = np.random.randn((N//K),D) + np.array([2,2])
X1 = np.random.randn((N//K),D) + np.array([0,-2])
X2 = np.random.randn((N//K),D) + np.array([-2,2])
X = np.vstack((X0,X1,X2))

y = np.array([0]*(N//K) + [1]*(N//K) + [2]*(N//K))


def one_hot_encode(y):
    N = len(y)
    K = len(set(y))

    Y = np.zeros((N,K))

    for i in range(N):
        Y[i,y[i]] = 1

    return Y

def shuffle(*args):
    idx = np.random.permutation(len(args[0]))
    return [X[idx] for X in args]

def ReLU(H):
    return H*(H>0)

def softmax(H):
    eH = np.exp(H)
    return eH / eH.sum(axis=1, keepdims=True)

def feed_forward(X,W1,b1,W2,b2):
    Z = ReLU(np.matmul(X,W1) + b1)
    P_hat = softmax(np.matmul(Z,W2) + b2)
    return Z, P_hat

def cross_entropy(Y,P_hat):
    return -np.sum(Y*np.log(P_hat))

def accuracy(y,P_hat):
    return np.mean(y == P_hat.argmax(axis=1))


M = 6

W1 = np.random.randn(D,M)
b1 = np.random.randn(M)
W2 = np.random.randn(M,K)
b2 = np.random.randn(K)

vW1 = 0
vb1 = 0
vW2 = 0
vb2 = 0

mu = 0.9

eta = 1e-3
epochs = int(1e3)
J = np.zeros(epochs)

for epoch in epochs:
    Z1,P_hat = feed_forward(X,W1,b1,W2,b2)
    J[epoch] = cross_entropy(Y,P_hat)

    dH2 = P_hat - Y
    dW2 = np.matmul(Z1.T,dH2)
    db2 = dH2.sum(axis=0)
    vW2 = mu*vW2 - eta*dW2
    vb2 = mu*vb2 - eta*db2
    W2 += vW2
    b2 += vb2

    dZ1 = np.matmul(dH2,W2.T)
    dH1 = dZ1 * (Z1 > 0)
    dW1 = np.matmul(X.T,dH1)
    db1 = dH1.sum(axis=0)
    vW1 = mu*vW1 - eta*dW1
    vb1 = mu*vb1 - eta*db1
    W1 += vW1
    b1 += vb1
