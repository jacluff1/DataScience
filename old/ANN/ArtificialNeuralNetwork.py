#===============================================================================
# import dependencies
#===============================================================================

# external
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# from my DataScience Library
import DataScience.ActivationFunctions as AF

#===============================================================================
# helper functions
#===============================================================================

def one_hot_encode(y):
    N,K = len(y),len(set(y))
    Y = np.zeros((N,K))
    for i in range(N):
        Y[i,y[i]] = 1
    return Y

#===============================================================================
# class definition
#===============================================================================

class ArtificialNeuralNetwork():

    def __init__(self,X,y,M,af,**kwargs):

        # check the input to make sure it is workable
        assert len(M) == len(af) - 1, "'M' must have one fewer entries than 'af'!"
        assert X.shape[0] == y.shape[0], "'X' and 'y' must have same number of observations!"

        # get dimentions
        self.N,self.D = X.shape
        self.L = len(af)
        self.hidden_layers = np.arange(2,self.L)

        # non standardized/normalized data
        self.X = X

        # non one hot encoded target array
        self.y = y

        self.Y

        # dictionary of nodes for each hidden layer
        self.M = M

        # a dictionary of ActivationFunction class instances for each layer
        self.af = af

        # make an shuffled indexing array
        self.__create_shuffle_index(**kwargs)

        # assign the number of observations for the train, validate, and test sets
        self.__assign_lengths_of_cv_sets(**kwargs)

    #===========================================================================
    # feed forward
    #===========================================================================

    def feed_forward_train(X,W,b,Gamma,mu,sigma2,af,**kwargs):
        raise NotImplemented

        # # dimentions
        # L = len(W)
        #
        # # kwargs
        # epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 1e-8
        #
        # # set collection dictionaries
        # H,Hbar,Z = {},{},{}
        #
        # # input
        # H[1] = np.matmul( X, W[1] )
        # Hbar[1] = Gamma[1] * ((H[1] - mu) / np.sqrt(sigma2 + epsilon)) + b[1]
        # Z[1] = af[1].f( Hbar[1] )
        #
        # # layers
        # for l in range(2,L+1):
        #     H[l] = np.matmul( Z[l-1], W[l] )
        #     Hbar[l] = Gamma[l] * ((H[l] - mu) / np.sqrt(sigma2 + epsilon)) + b[l]
        #     Z[l] = af[l].f( Hbar[l] )
        #
        # return H, Hbar, Z

    #===========================================================================
    # back propagation
    #===========================================================================

    #===========================================================================
    # cross validation
    #===========================================================================

    #===========================================================================
    # diagnostics
    #===========================================================================

    #===========================================================================
    # helper functions
    #===========================================================================

    def __create_shuffle_index(self,**kwargs):
        seed = kwargs['seed'] if 'seed' in kwargs else 0
        self.__shuffle_idx = np.random.RandomState(seed=seed).permutation(self.N)
        print(f"\nassigned shuffle index with seed {seed}")

    def __assign_lengths_of_cv_sets(self,**kwargs):
        train_fraction = kwargs['train_fraction'] if 'train_fraction' in kwargs else .6
        validation_fraction = kwargs['validation_fraction'] if 'validation_fraction' in kwargs else .2
        self.N_train = int(self.N * train_fraction)
        self.N_validate = int(self.N * validation_fraction)
        self.N_test = self.N - self.N_train - self.N_validate

    def __get_training_set(self):
        X,Y = self.X.copy(),self.Y.copy()
        idx = self.__shuffle_idx
        i_end = self.N_train
        return X[idx,:][:i_end,:], Y[idx,:][:i_end,:]

    def __get_validation_set(self):
        X,Y = self.X.copy(),self.Y.copy()
        idx = self.__shuffle_idx
        i_0 = self.N_train
        i_end = self.N_train + self.N_validate
        return X[idx,:][i_0:i_end,:], Y[idx,:][i_0:i_end,:]

    def __get_test_set(self):
        X,Y = self.X.copy(),self.Y.copy()
        idx = self.__shuffle_idx
        i_0 = self.N_train + self.N_validate
        return X[idx,:][i_0:,:], Y[idx,:][i_0:,:]

    

#===============================================================================
# example
#===============================================================================

def example(save_plot=False):

    # dimentions
    D = 2
    K = 3
    N = int(K*1e4)

    # data
    X0 = np.random.randn((N//K),D) + np.array([2,2])
    X1 = np.random.randn((N//K),D) + np.array([0,-2])
    X2 = np.random.randn((N//K),D) + np.array([-2,2])
    X = np.vstack((X0,X1,X2))

    # target
    y = np.array([0]*(N//K) + [1]*(N//K) + [2]*(N//K))

    # save plot
    if save_plot:
        plt.figure(figsize = (12,9))
        plt.scatter(X[:,0],X[:,1],c = y,alpha = 0.3333)
        plt.savefig("simulate_data_set.pdf")
        plt.close()

    # set up layers, nodes, and activating functions
    M = {1:4}
    af = {1:AF.ReLU(), 2:AF.softmax()}

    # create ArtificialNeuralNetwork instance
    ann = ArtificialNeuralNetwork(X, y, M, af, seed=1)

    # collect results
    results = {
        'X' :   X,
        'y' :   y,
        'D' :   D,
        'K' :   K,
        'N' :   N
        }

    # output
    return results
