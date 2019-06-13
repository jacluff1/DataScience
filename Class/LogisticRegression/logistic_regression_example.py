import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.py

D = 2
N = 1000

X0 = np.random.randn((N//2), D) + np.array([-1.5,1.5])
X1 = np.random.randn((N//2), D) + np.array([1.5,-1.5])
X = np.vstack( (X0,X1) )
X = np.column_stack( (np.ones((N,1)),X) )

y = np.array([0]*(N//2) + [1]*(N//2))

plt.figure()
plt.scatter(X[:,0], X[:,1], c=y)
plt.close()


def sigmoid(h):
    return 1/(1 + np.exp(-h))

def cross_entropy(y,p_hat):
    return -np.sum(y*np.log(p_hat) + (1-y)*np.log(1-p_hat))

def accuracy(y, p_hat):
    return np.mean(y == np.round(p_hat))

w = np.random.randn(D + 1)
eta = 1e-3
epochs = int(1e3)
J = [0]*epochs
for epoch in range(epochs):
    p_hat = sigmoid(X.dot(w))
    J[epoch] = cross_entropy(y,p_hat)
    w -= eta*X.T.dot(p_hat-y)

# plt.figure()
# plt.plot(J)
x1 = np.linspace(-5,5,50)
x2 = -(w[0]\w[2]) - (w[1]\w[2])*x1
plt.figure()
plt.scatter(X[:,1],X[:,2], c=y)
plt.plot(x1,x2, color="#000000", linewidth=2)


p_hat = sigmoid(X.dot(w))

print("Accuracy: {:0.4f}".format(accuracy(y,p_hat)))
