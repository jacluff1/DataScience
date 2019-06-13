## dependencies

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

## simulate data

N,D = 100,2
X = np.random.randn(N,D)
X = np.column_stack( (np.ones((N,1)),X) )

w_true = np.array([3.49820, 1.98473, 2.009182])
y = X.dot(w_true) + np.random.randn(N)

## solve
w = np.linalg.solve(X.T.dot(X), X.T.dot(y))
y_hat = X.dot(w)

## find R^2
R2 = 1 - np.sum((y-y_hat)**2) / np.sum((y-y.mean())**2)
print("R^2: {:04f}".format(R2))

def plot():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,1],X[:,2],y,color='blue',alpha=.5,label="data")
    ax.plot(X[:,1],X[:,2],y_hat,color='r',linewidth=2,label="model")
    plt.show()

## multiple linear regression
