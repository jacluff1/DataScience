# import external dependencies
import numpy as np
import pandas as pd

class Base:

    def __str__(self):
        pass

    #===========================================================================
    # basic
    #===========================================================================

    def predict(self,PHI):

        # check input
        assert hasattr(self,'W'), "Need to train model first dumb dumb!"
        assert PHI.shape[1] == self.W.shape[0], "Can't find Y_hat: PHI and W don't match!"

        # output
        return PHI.dot(self.W)

    #===========================================================================
    # normalize
    #===========================================================================

    def __normalize_data(self,PHI,PHI_train):

        # collect the min and maxes from PHI1 -- skipping the bias column
        xmin = PHI_train[:,1:].min(axis=0)
        xmax = PHI_train[:,1:].max(axis=0)

        # assign normalization mins and maxes to attributes
        self.xmin_normalization = xmin
        self.xmax_normalization = xmax
        print("\nadded 'xmin_normalization' and 'xmax_normalization' to attributes.")

        # make a copy of PHI and return the normalized copy
        PHI = PHI.copy()

        # normalize PHI against PHI1, skipping the bias column
        PHI[:,1] = (PHI[:,1] - xmin) / (xmax - xmin)

        # output
        return PHI

    def denormalize_weights(self):

        # alias xmin and xmax
        xmax = self.xmax_normalization
        xmin = self.xmin_normalization

        # reverse the normilzation process
        self.W[1:] = self.W[1:]*(xmax - xmin) + xmin

    #===========================================================================
    # cross validation
    #===========================================================================

    def cv_tvt_data_sets(self,PHI,Y,**kwargs):

        # kwargs
        assign = kwargs['assign'] if 'assign' in kwargs else True
        output = kwargs['output'] if 'output' in kwargs else False
        pickle = kwargs['pickle'] if 'pickle' in kwargs else False
        train_frac = kwargs['train_frac'] if 'train_frac' in kwargs else .60
        validate_frac = kwargs['validate_frac'] if 'validate_frac' in kwargs else .20
        normalize = kwargs['normalize'] if 'normalize' in kwargs else True

        # number of obseravation in entire set
        N = PHI.shape[0]

        # shuffle data
        PHI,Y = self.shuffle(PHI,Y)

        try:
            self.K = Y.shape[1]
        except:
            Y = self.one_hot_encode(Y)
            self.K = Y.shape[1]

        # get number of observations for each set
        N1 = int(N*train_frac)
        N2 = int(N*validate_frac)
        N3 = N - N1 - N2

        # split randomized observations into training (1), validation (2), and testing (3)
        PHI1 = PHI[:N1]
        PHI2 = PHI[N1:N1+N2]
        PHI3 = PHI[N1+N2:]
        Y1 = Y[:N1]
        Y2 = Y[N1:N1+N2]
        Y3 = Y[N1+N2:]

        # normalize data
        if normalize:
            PHI1 = self.__normalize_data(PHI1,PHI1)
            PHI2 = self.__normalize_data(PHI2,PHI1)
            PHI3 = self.__normalize_data(PHI3,PHI1)

        train = pd.Series( dict(PHI=PHI1, Y=Y1, normalized=normalize, N=PHI1.shape[0]) )
        validate = pd.Series( dict(PHI=PHI2, Y=Y2, normalized=normalize, N=PHI2.shape[0]) )
        test = pd.Series( dict(PHI=PHI3, Y=Y3, normalized=normalize, N=PHI3.shape[0]) )

        if assign:
            self.train = train
            self.validate = validate
            self.test = test
            self.D = PHI1.shape[1]
            print("\nassigned 'D', 'K', 'train', 'validate', and 'test' as attributes")

        if pickle:
            train.to_pickle('train.pkl')
            validate.to_pickle('validate.pkl')
            test.to_pickle('test.pkl')
            print("\npickled 'train.pkl', 'validate.pkl', and 'test.pkl'")

        if output: return train,validate,test

    #===========================================================================
    # helper functions
    #===========================================================================

    @staticmethod
    def shuffle(*args):
        idx = np.random.RandomState(seed=0).permutation(len(args[0]))
        return [X[idx] for X in args]

    @staticmethod
    def one_hot_encode(y):
        N = len(y)
        K = len(set(y))

        Y = np.zeros((N,K))

        for i in range(N):
            Y[i,y[i]] = 1

        return Y

    @staticmethod
    def softmax(H):
        eH = np.exp(H)
        return eH / eH.sum(axis=1, keepdims=True)

    def cross_entropy(self,Y):
        return -np.sum(Y*np.log(self.Z[self.L]))

    @staticmethod
    def accuracy(Y, P_hat):
        return np.mean(Y.argmax(axis=1) == P_hat.argmax(axis=1))

    def confusion_matrix(self,Y,P_hat):
        Y_hat = self.one_hot_encode(P_hat.argmax(axis=1))
        return np.matmul(Y.T,Y_hat)

    #===========================================================================
    # diagnostic - make a separte class for these?
    #===========================================================================

    # def plot_Y_vs_Y_hat(tvt_results):
    #     # make sure save directory exists and make filename
    #     if not os.path.isdir("check"): os.mkdir("check")
    #     filename = "Y_vs_Yhat.pdf"
    #     # make figure
    #     fig,ax = plt.subplots()
    #     fig.suptitle("Comparing Predicted Classification with Actual Classification", fontsize=20)
    #     ax.scatter(Y_hat,Y, color='r')
    #     ax.set_aspect(1)
    #     ax.set_xlabel("$\\hat{Y}", fontsize=15)
    #     ax.set_ylabel("Y", fontsize=15)
    #     fig.savefig(filename)
    #     print(f"\nsaved {filename}")
    #     plt.close(fig)

    # # nothing yet
    # def p_test(tvt_results,alpha=0.05):
    #     return
