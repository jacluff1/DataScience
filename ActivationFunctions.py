# import external dependencies
import numpy as np
import pdb

class softmax:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        eH = np.exp(H)
        assert np.all(np.isfinite(eH)), "NOOOOO! try decreasing learning rate."
        return eH / eH.sum(axis=1, keepdims=True)

    @ staticmethod
    def df(Hbar,**kwargs):
        return Hbar * (1 - Hbar)

class tanh:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        # eH = np.exp(H)
        # eHn = np.exp(-H)
        # return (eH - eHn) / (eH + eHn)
        return np.tanh(H)

    @staticmethod
    def df(Hbar,**kwargs):
        return 1 - Hbar**2

class ReLU:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        return H*(H > 0)

    @staticmethod
    def df(Hbar,**kwargs):
        return 1*(Hbar > 0)

class LReLU:

    def __str__(self):
        pass

    @staticmethod
    def f(H,**kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.01
        return H*(H >= 0) + H*alpha*(H < 0)

    @staticmethod
    def df(Hbar,**kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.01
        return 1*(Hbar >= 0) + alpha*(Hbar < 0)
