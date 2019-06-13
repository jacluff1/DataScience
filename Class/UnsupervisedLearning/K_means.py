import numpy as np
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm

class Kmeans():

    def __init__(self, X, K, beta):
        self.X = X
        self.K = K
        self.beta = beta
        self.N,self.D = X.shape
        idx = np.random.permutation(self.N)[:K]
        self.m = X.copy()[idx]

    #===========================================================================
    # distance
    #===========================================================================

    def euclidean(self):

class Kmeans1():

    def __init__(self, X, K, beta):
        self.X = X # data
        self.N,self.D = X.shape
        self.K = K # number of clusters
        self.y = np.zeros(self.N) # target
        self.beta = beta
        idx = np.random.permutation(self.N)[:K] # indicies for starting centroids
        self.m = X.copy()[idx] # centroids

    #===========================================================================
    # distance metrics
    #===========================================================================

    def euclidean(self,i,k):
        return np.sqrt( self.euclidean_squared(i,k) )

    def euclidean_squared(self,i,k):
        x_ik = self.X[i,:] - self.m[k,:]
        return x_ik.T @ x_ik

    #===========================================================================
    # hard
    #===========================================================================

    def mean(self):
        pass

    #===========================================================================
    # soft
    #===========================================================================

    def responsability(self):

        def numerator(i,k):
            return np.exp( -self.beta * self.euclidean_squared(i,k) )

        def denominator(i,k):
            return np.array([ np.exp(-self.beta * self.euclidean_squared(i,j)) for j in range(self.K) ]).sum()

        R = np.zeros((self.N,self.K))
        for i in range(self.N):
            for j in range(self.K):
                R[i,j] = numerator(i,j) / denominator(i,j)

        self.R = R

    def responsability_mean(self):
        m = np.zeros((self.K,self.D))
        for i in range(self.N):
            for k in range(self.K):
                m[k,:] += (self.R[i,k] * self.X[i,:]) / self.R[i,k]
        return m

    def distortion(self):
        total = 0
        for i in range(self.N):
            for j in range(self.K):
                total += self.R[i,j] * self.euclidean_squared(self.X[i] - self.m[k])
        return total

    #===========================================================================
    # fit
    #===========================================================================

    def find_clusters(self):
        self.y = self.R.argmax(axis=1)

    def fit(self,max_iterations, threshold=1e-3):

        for _ in tqdm(range(max_iterations)):

            # update responsability
            self.responsability()

            # update mean
            m = self.responsability_mean()
            # dist = np.sqrt( np.sum( (self.m - m)**2 , axis=1 ) )
            # if np.all(dist < 1e-3): break
            self.m = m
            print(m)

#===============================================================================
# example
#===============================================================================

N,K,D = int(1e3),3,2
beta = 1

def example():

    X0 = np.random.randn(N//K,D) + np.array([2,2])
    X1 = np.random.randn(N//K,D) + np.array([2,-2])
    X2 = np.random.randn(N//K,D) + np.array([-2,0])
    X = np.vstack((X0,X1,X2))

    kmeans = Kmeans(X, K, beta)
    kmeans.fit(30)
    m = kmeans.m

    fig = plt.figure(figsize=(10,10))
    plt.scatter(X[:,0],X[:,1], c=kmeans.y, alpha=.5)
    plt.scatter(m[:,0],m[:,1], c='r', s=100)
    plt.show()

if __name__ == "__main__":
    example()
