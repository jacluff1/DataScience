# import external dependencies
import numpy as np
import pdb

class softmax:

    def __str__(self):
        pass

    @staticmethod
    def f(*args,**kwargs):
        H = args[0]
        eH = np.exp(H)
        return eH / eH.sum(axis=1, keepdims=True)

    @ staticmethod
    def df(*args,**kwargs):
        H = args[0]
        eH = np.exp(H)
        P = eH / eH.sum(axis=1, keepdims=True)
        return P * (1 - P)

class tanh:

    def __str__(self):
        pass

    @staticmethod
    def f(*args,**kwargs):
        H = args[0]
        eH = np.exp(H)
        eHn = np.exp(-H)
        return (eH - eHn) / (eH + eHn)

    @staticmethod
    def df(*args,**kwargs):
        H = args[0]
        eH = np.exp(H)
        eHn = np.exp(-H)
        P = (eH - eHn) / (eH + eHn)
        return 1 - P**2

class ReLU:

    def __str__(self):
        pass

    @staticmethod
    def f(*args,**kwargs):
        H = args[0]
        return H*(H > 0)

    @staticmethod
    def df(*args,**kwargs):
        dZ = args[0],
        Z = args[1]
        return dZ*(Z > 0)

class LReLU:

    def __str__(self):
        pass

    @staticmethod
    def f(*args,**kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.01
        H = args[0]
        return H*(H >= 0) + H*alpha*(H < 0)

    @staticmethod
    def df(*args,**kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.01
        dZ = args[0]
        Z = args[1]
        return dZ*(Z >= 0) + alpha*dZ*(Z < 0)
