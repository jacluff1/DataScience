# import external dependencies
import numpy as np

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

    def _denormalize_weights(self):

        # alias xmin and xmax
        xmax = self.xmax_normalization
        xmin = self.xmin_normalization

        # reverse the normilzation process
        self.W[1:] = self.W[1:]*(xmax - xmin) + xmin

    #===========================================================================
    # cross validation
    #===========================================================================

    def _cv_tvt_pickle_sets(self,**kwargs):

        # kwargs
        train_frac = kwargs['train_frac'] if 'train_frac' in kwargs else .60
        validate_frac = kwargs['validate_frac'] if 'validate_frac' in kwargs else .20
        normalize = kwargs['normalize'] if 'normalize' in kwargs else True

        # start with the initial data
        PHI = self.PHI.copy()
        Y = self.Y.copy()

        # get mask to shuffle for train-validate-test
        mask = np.arange(self.N)
        np.random.shuffle(mask)
        PHI0,Y0 = PHI[mask],Y[mask]

        # get number of observations for each set
        N1 = int(self.N*train_frac)
        N2 = int(self.N*validate_frac)
        N3 = self.N - N1 - N2

        # split randomized observations into training (1), validation (2), and testing (3)
        PHI1 = PHI0[:N1]
        PHI2 = PHI0[N1:N1+N2]
        PHI3 = PHI0[N1+N2:]
        Y1 = Y0[:N1]
        Y2 = Y0[N1:N1+N2]
        Y3 = Y0[N1+N2:]

        # normalize data
        if normalize:
            PHI1 = self._normalize_data(PHI1,PHI1)
            PHI2 = self._normalize_data(PHI2,PHI1)
            PHI3 = self._normalize_data(PHI3,PHI1)

        # # add results as attributes
        # self.train = dict(PHI=PHI1, Y=Y1, normalized=normalize)
        # self.validate = dict(PHI=PHI2, Y=Y2, normalized=normalize)
        # self.test = dict(PHI=PHI3, Y=Y3, normalized=normalize)

        pd.Series( dict(PHI=PHI1, Y=Y1, normalized=normalize) ).to_pickle('./train.pkl')
        pd.Series( dict(PHI=PHI2, Y=Y2, normalized=normalize) ).to_pickle('./validate.pkl')
        pd.Series( dict(PHI=PHI3, Y=Y3, normalized=normalize) ).to_pickle('./test.pkl')
        print("\nsaved 'train.pkl', 'validate.pkl', and 'test.pkl'")

    #===========================================================================
    # diagnostic
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

    # nothing yet
    def p_test(tvt_results,alpha=0.05):
        return
