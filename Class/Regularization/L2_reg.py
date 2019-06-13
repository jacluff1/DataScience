# import dependencies
import numpy as np
import matplotlib.pyplot as plt

# simulate data
N = 30
x = np.linspace(0,10,N)
y = 2.87403 + 0.8298*x + np.random.randn(N)
# insert outliers
y[(N-2):] += 20

X = np.column_stack( (np.ones(N),x) )
w = np.linalg.solve(X.T.dot(X), X.T.dot(y))
y_hat = X.dot(w)

I_reg = np.identity(2)
# I_reg[0,0] = 0
lambda2 = 500
w_l2 = np.linalg.solve( X.T.dot(X) + lambda2*I_reg, X.T.dot(y) )
y_hat_l2 = X.dot(w_l2)

def OLS(y,y_hat):
    return (y-y_hat).dot(y-y_hat)

def r_squared(y,y_hat):
    return 1 - np.sum( (y-y_hat)**2 ) / np.sum( (y-y.mean())**2 )

# w = np.random.randn(2)
w = np.random.randn(1)
b = np.random.randn(1)
J = []
eta = 1e-3
epochs = int(1e3)
lambda2 = 500

for epoch in range(epochs):
    y_hat = X.dot(w)
    # J.append(OLS(y,y_hat) + lambda2*w.dot(w))
    # w_reg = w
    # w_reg[0] = 0
    # w -= eta*(X.T.dot(y_hat-y) + lambda2*w_reg)
    J.append(OLS(y,y_hat) + lambda2*w**2)
    w -= eta * (x.dot(y_hat-y) + lambda2*w)
    b -= eta * (y_hat - y).sum()

plt.figure()
# plt.plot(J)
plt.scatter(x,y, color='b', label="data")
plt.plot(x,y_hat, color='r', linewidth=2, label="non regularized")
plt.plot(x,y_hat_l2, color='orange', linewidth=2, label="regularized")
plt.xlim(x.min(),x.max())
plt.legend(loc='best')
plt.show()
