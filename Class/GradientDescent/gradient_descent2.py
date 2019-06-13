import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, laplace

N = 50
D = 50

X = np.random.randn(N,D)
X = np.column_stack((np.ones((N,1)),X))
w_true = np.array([2.974, 8.298, 1.23748] + [0]*(D-2))
y = X.dot(w_true) + np.random.randn(N)

def OLS(y,y_hat):
    return (y-y_hat).dot(y-y_hat)

def r_squared(y,y_hat):
    return 1 - np.sum( (y-y_hat)**2 ) / np.sum( (y-y.mean())**2 )

w = np.random.randn(D+1)
J = []
eta = 1e-3
epochs = int(1e3)
for epoch in range(epochs):
    y_hat = X.dot(w)
    J.append(OLS(y,y_hat))
    w -= eta*X.T.dot(y_hat-y)

# plt.figure()
# plt.scatter(w_true, label="True weights")
# plt.scatter(w, label="Estimated weights")
# plt.plot(J)
# plt.show()

w_l1 = np.random.randn(D+1)
J = []
eta = 1e-3
epochs = int(1e3)
lambda1 = 10

for epoch in range(epochs):
    y_hat = X.dot(w_l1)
    J.append(OLS(y,y_hat) + lambda1*np.sum(np.abs(w_l1)))
    w_l1 -= eta * (X.T.dot(y_hat-y) + lambda1*np.sign(w_l1))

# plt.figure()
# plt.plot(w_true, label="True weights")
# plt.plot(w, label="Estimated weights")
# plt.plot(w_l1, label="LASSO weights")
# plt.legend()
# # plt.plot(J)
# plt.show()

x = np.linspace(-10,10,101)
lambda1 = 1
lambda2 = .5
p_l1 = laplace.pdf(x,0,1/lambda1)
p_l2 = multivariate_normal.pdf(x,0,1/lambda2)

plt.figure()
# plt.plot(x,p_l1, label="Laplace")
# plt.plot(x,p_l2, label="Gaussian")
plt.plot(x,p_l1*p_l2, label="Laplaussian")
plt.legend()
plt.show()
