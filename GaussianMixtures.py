# import dependencies
import numpy as np
from scipy.stats import multivariate_normal as normal
import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import pdb
import matplotlib.pyplot as plt

class GaussianMixtureModel:

    def __init__(self, X, **kwargs):
        self.X = X
        self.N,self.D = X.shape
        self.set_options(**kwargs)

    def set_options(self, **kwargs):
        self.epochs = kwargs['epochs'] if 'epochs' in kwargs else 30
        self.threshold = kwargs['threshold'] if 'threshold' in kwargs else 1e-2
        self.early_break = kwargs['early_break'] if 'early_break' in kwargs else True

    def difference_vectors(self):
        # K x N x D
        self.DIFF = self.X - self.mu.reshape([ self.K , 1 , self.D ])

    def gaussian(self, K):
        # K x N
        pdb.set_trace()
        self.Gauss = np.array([ normal.pdf(self.X, mean=self.mu[k,:], cov=self.Sigma[k,:,:]) for k in range(self.K) ]) / np.sqrt( (2*np.pi)**self.K * np.linalg.inv(self.Sigma) )

    # def responsibility(self):
    #     # K x N
    #     self.r = self.pi * self.Gauss / (self.pi * self.Gauss).sum(axis=0)
    #
    # def mean(self):
    #     self.mu = (self.r * self.X).sum(axis=1) / self.r.sum(axis=1, keepdims=True)
    #
    # def covariance(self, K):
    #
    #     def num(k):
    #         v = self.X[i,:] - self.mu[k,:]
    #         return np.array([ self.r[:,i] * (v @ v.T) for i in range(self.N) ])
    #
    #     den = self.r.sum(axis=1)
    #
    #     self.Sigma = ( (self.X-self.mu)@(self.X-self.mu)      num = (self.pi * np.array([ multivariate_normal(self.X, self.mu[k,:], self.Sigma[k,:,:]) for k in range(K) ])).T ).sum(axis=0)
    #
    # def pi_k(self):
    #     self.pi = self.r.sum(axis=1) / self.N

    def fit(self, K):

        # assign starting mean, covariance, & pi at random
        self.mu = np.random.randn(self.K,self.D)
        self.Sigma = np.random.randn(self.K,self.D,self.D) + 1e-8 * np.eye(D)
        self.pi = np.random.randn(self.K)

        J = []

        for epoch in range(self.epochs):

            self.responsibility()
            self.mean()
            self.covariance()
            self.pi_k()

if "__name__" == "__main__":

    df = pd.read_csv("data/gmm.csv", names=['x', 'y'], header=None)
    gmm = GaussianMixtureModel(df.values)

    X = gmm.X
    y = np.argmax(gmm.responsibility, axis=0)
