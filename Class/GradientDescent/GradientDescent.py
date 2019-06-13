# import dependencies
import numpy as np
import matplotlib.pyplot as plt

# simulate data
N = 100
x = np.linspace(0,10,N)
y = 7.1234 + 0.74873*x + np.random.randn(N)

# data prep
X = np.vstack( (np.ones(N),x) ).T

def OLS(y,y_hat):
    return (y-y_hat).dot(y - y_hat)

def r_squared(y,y_hat):
    return 1 - np.sum( (y-y_hat)**2 ) / np.sum( (y-y.mean())**2 )

# fit model using gradient descent
w = np.random.randn(X.shape[1])
J = []
eta = 1e-4
epochs = int(2e3)

for epoch in range(epochs):
    y_hat = X.dot(w)
    J.append(OLS(y,y_hat))
    w -= eta*X.T.dot(y_hat-y)

y_hat = X.dot(w)

# plt.figure()
# plt.plot(J)
# plt.show()

# plot
plt.figure()
plt.scatter(x,y)
plt.plot(x,y_hat,color='red',linewidth=2)
plt.show()
