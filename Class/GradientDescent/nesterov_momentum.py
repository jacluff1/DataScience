import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
Y = one_hot_encode(y)

W1 = np.random.randn(D,M)
b1 = np.random.randn(M)
W2 = np.random.randn(M,K)
b2 = np.random.randn(K)

adam = {'v1':0, 'vb1':0, 'vW1':0, 'vb2':0, 'W1':W1.copy(), 'b1':b1.copy(), 'W2':W2.copy(), 'b2':b2.copy(), 'mb1':0, 'mb2':0, 'mW1':0, 'mW2':0, 'dH2':0, 'dH1':0, 'dW2':0, 'dW1':0, 'db2':0, 'db1':0, 'dZ1':0, 'vW2':0}
rms = {'v1':0, 'vb1':0, 'vb2':0, 'vW1':0, 'vW2':0, 'W1':W1.copy(), 'b1':b1.copy(), 'W2':W2.copy(), 'b2':b2.copy(), 'dH2':0, 'dH1':0, 'dW2':0, 'dW1':0, 'db2':0, 'db1':0, 'dZ1':0}

mu = 0.9
gamma = 0.999
epsilon = 1e-8
G2W = 0
G1W = 0
G2b = 0
G1b = 0

eta = 1e-8
epochs = int(1e3)
J_rms = np.zeros(epochs)
J_adam = np.zeros(epochs)

for epoch in range(1,epochs):

    Z1_adam,P_hat_adam = feed_forward(X,adam['W1'],adam['b1'],adam['W2'],adam['b2'])
    Z1_rms,P_hat_rms = feed_forward(X,rms['W1'],rms['b1'],rms['W2'],rms['b2'])
    J_rms[epoch] = cross_entropy(Y,P_hat_rms)
    J_adam[epoch] = cross_entropy(Y,P_hat_adam)

    #RMSprop
    rms['dH2'] = P_hat_rms - Y
    rms['dW2'] = np.matmul(Z1_rms.T,rms['dH2'])
    rms['db2'] = rms['dH2'].sum(axis=0)
    G2W = gamma*G2W + (1-gamma)*rms['dW2']**2
    G2b = gamma*G2b + (1-gamma)*rms['db2']**2
    rms['vW2'] = mu*rms['vW2'] - (eta/np.sqrt(G2W + epsilon))*rms['dW2']
    rms['vb2'] = mu*rms['vb2'] - (eta/np.sqrt(G2b + epsilon))*rms['db2']
    rms['W2'] -= (eta/np.sqrt(G2W + epsilon)) * rms['dW2']
    rms['b2'] -= (eta/np.sqrt(G2b + epsilon)) * rms['db2']
    # Adam
    adam['dH2'] = P_hat_adam - Y
    adam['dW2'] = np.matmul(Z1_adam.T,adam['dH2'])
    adam['db2'] = rms['dH2'].sum(axis=0)
    adam['mW2'] = (mu*adam['mW2'] + (1-mu)*adam['dW2'])
    adam['mb2'] = (mu*adam['mb2'] + (1-mu)*adam['db2'])
    adam['vW2'] = (mu*adam['vW2'] + (1-gamma)*adam['dW2']**2)
    adam['vb2'] = (mu*adam['vb2'] + (1-gamma)*adam['db2']**2)
    adam['W2'] -= (eta/np.sqrt(adam['vW2']/(1 - gamma**epoch) + epsilon)) * adam['mW2']/(1 - mu**epoch)
    adam['b2'] -= (eta/np.sqrt(adam['vb2']/(1 - gamma**epoch) + epsilon)) * adam['mb2']/(1 - mu**epoch)

    #RMSprop
    rms['dZ1'] = np.matmul(rms['dH2'],rms['W2'].T)
    rms['dH1'] = rms['dZ1'] * (Z1_rms > 0)
    rms['dW1'] = np.matmul(X.T,rms['dH1'])
    rms['db1'] = rms['dH1'].sum(axis=0)
    G1W = gamma*G1W + (1-gamma)*rms['dW1']**2
    G1b = gamma*G1b + (1-gamma)*rms['db1']**2
    rms['vW1'] = mu*rms['vW1'] - (eta/np.sqrt(G1W + epsilon))*rms['dW1']
    rms['vb1'] = mu*rms['vb1'] - (eta/np.sqrt(G1b + epsilon))*rms['db1']
    rms['W1'] -= (eta/np.sqrt(G1W + epsilon)) * rms['dW1']
    rms['b1'] -= (eta/np.sqrt(G1b + epsilon)) * rms['db1']
    # Adam
    adam['dZ1'] = np.matmul(adam['dH2'],adam['W2'].T)
    adam['dH1'] = adam['dZ1'] * (Z1_adam > 0)
    adam['dW1'] = np.matmul(X.T,adam['dH1'])
    adam['db1'] = adam['dH1'].sum(axis=0)
    adam['mW1'] = (mu*adam['mW1'] + (1-mu)*adam['dW1'])
    adam['mb1'] = (mu*adam['mb1'] + (1-mu)*adam['db1'])
    adam['vW1'] = (mu*adam['vW1'] + (1-gamma)*adam['dW1']**2)
    adam['vb1'] = (mu*adam['vb1'] + (1-gamma)*adam['db1']**2)
    adam['W1'] -= (eta/np.sqrt(adam['vW1']/(1 - gamma**epoch) + epsilon)) * adam['mW1']/(1 - mu**epoch)
    adam['b1'] -= (eta/np.sqrt(adam['vb1']/(1 - gamma**epoch) + epsilon)) * adam['mb1']/(1 - mu**epoch)

plt.figure()
plt.plot(J_rms, label="RMSprop with Momentum")
plt.plot(J_adam, label="Adam")
plt.legend()
plt.savefig("J.pdf")
plt.close()
