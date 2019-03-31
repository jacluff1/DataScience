# import external
import numpy as np

class Layer:

    def __init__(self,activation_class):
        """
        activation_class: class instance of an activation function
        """
        # assign the activation function class to the instance
        self.af = activation_class
        # add important values to instance with default None values
        self.M = None # number of nodes in layer
        self.Z = None # output data matrix of layer
        self.H = None # prediction matrix/logit
        self.W = None # weight matrix
        self.b = None # bias vector
        self.gradient_Z = None # gradient of objective function WRT Z
        self.gradient_H = None # gradient of objective function WRT H
        self.gradient_W = None # gradient of objective function WRT W

    def __str__(self):
        pass

    def feed_forward_H(self,X):
        self.H = np.matmul(X,self.W) + self.b


    def feed_forward_Z(self,**kwargs):
        self.Z = self.af.f(self.H,**kwargs)
