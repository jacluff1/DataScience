# import external dependencies
import numpy as np
import pdb

class softmax:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        eH = np.exp(H)
        return eH / eH.sum(axis=1, keepdims=True)

    @ staticmethod
    def df(H,**kwargs):
        eH = np.exp(H)
        P = eH / eH.sum(axis=1, keepdims=True)
        return P * (1 - P)

class tanh:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        eH = np.exp(H)
        eHn = np.exp(-H)
        return (eH - eHn) / (eH + eHn)

    @staticmethod
    def df(H,**kwargs):
        eH = np.exp(H)
        eHn = np.exp(-H)
        P = (eH - eHn) / (eH + eHn)
        return 1 - P**2

class ReLU:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        return H*(H > 0)

    @staticmethod
    def df(H,**kwargs):
        return 1*(H > 0)

class LReLU:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.01
        # pdb.set_trace()
        return H*(H >= 0) + H*alpha*(H < 0)

    @staticmethod
    def df(H,**kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.01
        return 1*(H >= 0) + alpha*(H < 0)
