# import external dependencies
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb
# import internal dependencies
from DataScience.Layer import Layer

#===============================================================================
# functions
#===============================================================================

def one_hot_encode(y):
    N = len(y)
    K = len(set(y))

    Y = np.zeros((N,K))

    for i in range(N):
        Y[i,y[i]] = 1

    return Y

def cross_entropy(Y,P_hat):
    return -np.sum(Y*np.log(P_hat))

def accuracy(Y, P_hat):
    return np.mean(Y.argmax(axis=1) == P_hat.argmax(axis=1))

#===============================================================================
# ANN class
#===============================================================================

class ArtificalNeuralNet:

    def __init__(self,X,Y,activation_functions,Ms):
        """
        activation_functions:
            dict of activation class instances with int keys that correspond to the layer in which they are applied. 1 <= key <= L
        Ms:
            dict of int values for the length of each layer. the keys correspond to the layer in which they are applied. 1 <= key <= L-1
        """

        #=======================================================================
        # basic set up
        #=======================================================================

        # assign input data to instance
        self.X = X

        # get number of layers
        self.L = len(activation_functions)

        # self.af = activation_functions

        # hidden layer indicies
        self.__hl_keys = np.arange(2,self.L)

        # save node numbers
        self.M = Ms

        # save dimentions from input data
        self.D = X.shape[1]
        if Y.shape[0] == Y.size:
            self.K = 1
        else:
            self.K = Y.shape[1]

        #=======================================================================
        # instantiate layers
        #=======================================================================

        self.layers = {}
        for l in range(1,self.L+1):
            self.layers[l] = Layer(activation_functions[l])

        # set random Ws and bs for first layer
        self.layers[1].W = np.random.randn(self.D,Ms[1])
        self.layers[1].b = np.random.randn(Ms[1])

        # set up random Ws and bs for hidden layers
        for l in self.__hl_keys:
            self.layers[l].W = np.random.randn(Ms[l-1],Ms[l])
            self.layers[l].b = np.random.randn(Ms[l])

        # set up random Ws and bs for output layer
        self.layers[self.L].W = np.random.randn(Ms[self.L-1],self.K)
        self.layers[self.L].b = np.random.randn(self.K)

    def __str__(self):
        pass

    #===========================================================================
    # feed forward
    #===========================================================================

    def feed_forward(self,X,**kwargs):

        self.layers[1].feed_forward_H(X)
        self.layers[1].feed_forward_Z(**kwargs)

        for l in range(2,self.L+1):
            self.layers[l].feed_forward_H(self.layers[l-1].Z)
            self.layers[l].feed_forward_Z(**kwargs)

    #===========================================================================
    # back propagation
    #===========================================================================

    def __gradient_Z(self,l):
        gZ = np.matmul(self.layers[l+1].gradient_H,self.layers[l+1].W.T)
        self.layers[l].gradient_Z = gZ

    def __gradient_H(self,l,**kwargs):
        # pdb.set_trace()
        gH = self.layers[l].gradient_Z * self.layers[l].af.df(self.layers[l].H,**kwargs)
        self.layers[l].gradient_H = gH

    def __gradient_W(self,l,**kwargs):
        gW = np.matmul( self.layers[l-1].Z.T , self.layers[l].gradient_H )
        self.layers[l].gradient_W = gW

    def back_propagation(self,X,Y,**kwargs):

        # kwargs
        plot = kwargs['eta'] if 'eta' in kwargs else True
        eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
        epochs = int(epochs)
        N = X.shape[0]

        J = np.zeros(epochs)

        for epoch in range(epochs):

            # feed forward
            self.feed_forward(X,**kwargs)
            P_hat = self.layers[self.L].Z
            J[epoch] = cross_entropy(Y,P_hat)

            # start journey back to X at the output layer
            self.layers[self.L].gradient_H = P_hat - Y
            # self.__gradient_H(self.L,**kwargs)
            self.__gradient_W(self.L,**kwargs)
            dW = self.layers[self.L].gradient_W
            db = self.layers[self.L].gradient_H.sum(axis=0)
            self.layers[self.L].W -= eta*dW
            self.layers[self.L].b -= eta*db

            # continue back through all the hidden layers
            for l in self.__hl_keys[::-1]:
                self.__gradient_Z(l)
                self.__gradient_H(l,**kwargs)
                self.__gradient_W(l,**kwargs)
                dW = self.layers[l].gradient_W
                db = self.layers[l].gradient_H.sum(axis=0)
                self.layers[l].W -= eta*dW
                self.layers[l].b -= eta*db

        # print accuracy
        print("Accuracy: {:0.4f}".format(accuracy(Y,P_hat)))

        # plot
        if plot:
            plt.figure()
            plt.plot(J)
            plt.title(f"$\\eta$: {eta}, epochs: {epochs}")
            plt.savefig(f"J/J_eta_{eta}_epochs_{epochs}.pdf")
            plt.close()
