import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def OLS(Y,Y_hat):
    return np.sum( (Y-Y_hat)**2 )

def r_squared(Y,Y_hat):
    return 1 - np.sum( (Y-Y_hat)**2 ) / np.sum( (Y-Y.mean())**2 )

class LinearRegression:

    def __str__(self):
        pass

    def fit(self, X, Y, eta=1e-3, lambda1=0, lambda2=0, epochs=1e3, show_curve=False):
        N,D = X.shape
        if len(Y.shape) > 1:
            K = Y.shape[1]
        else:
            K = 1

        self.W = np.random.randn(D,K)

        epochs = int(epochs)
        J = np.zeros(epochs)

        for epoch in range(epochs):
            Y_hat = self.predict(X)
            J[epoch] = OLS(Y, Y_hat) + lambda1*np.sum(np.abs(self.W)) + lambda2*np.sum(self.W**2)
            self.W -= eta*(X.T.dot(Y_hat-Y) + lambda1*np.sign(self.W) + lambda2*self.W)

        if show_curve:
            plt.figure()
            plt.plot(J)
            plt.xlabel("epochs")
            plt.ylabel("J")
            plt.savefig("J.pdf")
            plt.close()

    def predict(self, X):
        return X.dot(self.W)

lm = LinearRegression(lm)
