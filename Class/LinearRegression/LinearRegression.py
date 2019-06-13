## import dependencies

import numpy as np
import matplotlib.pyplot as plt

    def denominator(X,Y):
        return np.average(X**2) - np.average(X)**2

    def intercept(X,Y,den):
        num =  np.average(Y) * np.average(X**2) - np.average(X) * np.average(X*Y)
        return num/den

    def slope(X,Y,den):
        num = np.average(X*Y) - np.average(X) * np.average(Y)
        return num/den

    def R2(Y,y_hat):
        return 1 - np.sum( (Y - y_hat)**2 ) / np.sum( (Y - Y.mean())**2 )

    def plot(X,Y):

        fig = plt.figure()
        plt.scatter(X,Y)

        den = denominator(X,Y)
        w = self.slope(X,Y,den)
        b = self.intercept(X,Y,den)
        y_hat = w*X + b
        plt.plot(X,y_hat, color='red', linewidth=2)
        plt.show()

## simulate data
N = 100
X = np.linspace(0,10,N)
Y = 3.276 * X + 4.907531 + np.random.randn(N)

LR = LinearRegression(X,Y)
LR.plot(X,Y)
