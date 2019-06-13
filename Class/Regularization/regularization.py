import numpy as np
import matplotlib.pyplot

N = 103
D = 2

X = np.random.randn(N,D)
X = np.column_stack( (np.ones((N,1)), X) )
w_true = np.array([3.485, 298048, 1.04957])
y = X.dot(w_true) + np.random.randn(N)

w = np.linalg.solve(X.T.dot(X), X.T.dot(y))
y_hat = X.dot(w_true)

def r_squared(y,y_hat):
    R2 = 1 - np.sum( (y-y_hat)**2 ) / np.sum( (y-y.mean())**2 )
    print("R-squared w/ two vars: {}".format(R2))
    return R2

X2 = np.column_stack((X,np.random.randn((N,1))))
w2 = np.linalg.solve(X2.T.dot(X2), X2.T.dot(y))
y_hat2 = X2.dot(w2)
r22 = r_squared(y,y_hat2)
