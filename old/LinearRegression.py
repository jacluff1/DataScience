# import external libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import from internal modules
from DataScience.Base import Base

#===============================================================================
# functions outside of class
#===============================================================================

def OLS(Y,Y_hat):

    # check input
    assert Y.shape == Y_hat.shape, "Can't find OLS: Y and Y_hat don't match!"

    # output
    return (Y-Y_hat).dot(Y-Y_hat)

def r_squared(Y,Y_hat):

    # check input dimentions
    assert Y.shape == Y_hat.shape, "Can't find R^2: Y and Y_hat don't match!"

    # ouput
    return 1 - np.sum( (Y-Y_hat)**2 ) / np.sum( (Y-Y.mean())**2 )

#===============================================================================
# class
#===============================================================================

class LinearRegression(Base):

    def __init__(self,PHI,Y):

        # add input as attributes
        self.PHI = PHI
        self.Y = Y

        # add some basic info on data
        self.N = PHI.shape[0]
        self.P = PHI.shape[1]
        try:
            self.K = Y.shape[1]
        except:
            self.K = 1

        print("added 'PHI', 'Y', 'N', 'P', and 'K' to attributes.")

    def __str__(self):
        pass

    #===========================================================================
    # basic methods
    #===========================================================================

    def predict(self,PHI):

        # check input
        assert hasattr(self,'W'), "Need to train model first dumb dumb!"
        assert PHI.shape[1] == self.W.shape[0], "Can't find Y_hat: PHI and W don't match!"

        # output
        return PHI.dot(self.W)

    #===========================================================================
    # solve
    #===========================================================================

    def __solve_gradient_descent(self,PHI,Y,**kwargs):

        # kwargs
        lambda1 = kwargs['lambda1'] if 'lambda1' in kwargs else 0
        lambda2 = kwargs['lambda2'] if 'lambda2' in kwargs else 0
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
        eta = kwargs['eta'] if 'eta' in kwargs else 1e-3

        # make empty list for OLS
        epochs = int(epochs)
        J = np.zeros(epochs)

        # start with random W
        if self.K > 1:
            self.W = np.random.randn(self.P*self.K).reshape([self.P,self.K])
        else:
            self.W = np.random.randn(self.P)

        # try and solve for w
        for epoch in range(epochs):
            Y_hat = self.Y_hat(PHI)
            J[epoch] = ( OLS(Y,Y_hat)/2 + lambda1*np.sum(np.abs(self.W)) + lambda2*np.sum(self.W**2)/2 )/N
            self.W -= eta*(PHI.T.dot(Y_hat-Y) + lambda1*np.sign(self.W) + lambda2*self.W)/N

        return J

    # add more numerical methods as we learn them

    def __solve_numerically(self,PHI,Y,**kwargs):

        # kwargs
        method = kwargs['method'] if 'method' in kwargs else "gradient descent"
        save_curve = kwargs['save_curve'] if 'save_curve' in kwargs else False

        # select solving method
        if method == "gradient_descent":
            J = self.__solve_gradient_descent(PHI,Y,**kwargs)

        # save plot
        if save_curve:
            fig = plt.figure()
            plt.plot(J)
            plt.title(f"$\\lambda_1$: {lambda1}, $\\lambda_2$: {lambda2}, $\\eta$: {eta}, epochs: {epochs}")
            filename = f"L1_{lambda1}_L2_{lambda2}_epochs_{epochs}_eta_{eta}.pdf"
            fig.savefig(filename)
            print(f"\nsaved {filename}")
            plt.close(fig)

    def __solve_analytically(self,PHI,Y,**kwargs):

        # kwargs
        lambda2 = kwargs['lambda2'] if 'lambda2' in kwargs else 0

        # L2 regularization
        if lambda2 > 0:
            I = np.eye(PHI.shape[1])
            I[0,0] = 0
            self.W = np.linalg.solve(PHI.T.dot(PHI) + lambda2*I, PHI.T.dot(Y))
        else:
            # No Regularization
            self.W = np.linalg.solve(PHI.T.dot(PHI), PHI.T.dot(Y))

    def solve(self,PHI,Y,**kwargs):

        # check input
        assert PHI.shape[0] == Y.shape[0], "Can't find closed form solution: PHI and Y don't match!"

        # kwargs
        output = kwargs['output'] if 'output' in kwargs else False
        lambda1 = kwargs['lambda1'] if 'lambda1' in kwargs else 0
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
        epochs = int(epochs)
        N = PHI.shape[0]

        # check best solution method
        if any([ lambda1 > 0 , N*epochs < self.P**3 ]):
            self.__solve_numerically(PHI,Y,**kwargs)
        else:
            self.__solve_analytically(PHI,Y,**kwargs)

        # output
        if output: return self.W

    #===========================================================================
    # cross validation
    #===========================================================================

    def cv_train_validate_test(self,L1,L2,**kwargs):

        # make sure train, validate, and test groups have been set up
        if not hasattr(self,'train'): self.__cv_tvt_add_sets(**kwargs)

        # kwargs
        output = kwargs['output'] if 'output' in kwargs else False
        pickle = kwargs['pickle'] if 'pickle' in kwargs else False
        add_to_self = kwargs['add_to_self'] if 'add_to_self' in kwargs else True
        eata = kwargs['eta'] if 'eta' in kwargs else 1e-3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
        epochs = int(epochs)

        # set up dictionary to collect results
        results = dict(eta=eta, epochs=epochs)
        bestR2_validate = 0

        # cast a net and find best outcome
        for l1 in L1:
            for l2 in L2:

                # print iteration to track progress
                print(f"\nworking on {l1},{l2}...")

                # update kwargs
                kwargs['lambda1'] = l1
                kwargs['lambda2'] = l2

                # train
                self.solve(self.train['PHI'],self.train['Y'],**kwargs)
                Y_hat_train = self.predict(self.train['PHI'])
                R2_train = self.r_squared(self.train['Y'],Y_hat_train)

                # validate
                Y_hat_validate = self.predict(self.validate['PHI'])
                R2_validate = self.r_squared(self.validate['Y'],Y_hat_validate)

                # print iteration results
                print(f"train R-squared: {R2_train}, validate R-squared {R2_validate}")

                # update best values
                if R2_validate > bestR2_validate:

                    # run the test set
                    Y_hat_test = self.predict(self.test['PHI'])
                    R2_test = self.r_squared(self.test['Y'],Y_hat_test)

                    # collect the best values
                    results['lambda1'] = l1
                    results['lambda2'] = l2
                    results['R2'] = dict(train=R2_train, validate=R2_validate, test=R2_test)
                    results['W'] = self.W.copy()
                    bestR2_validate = R2_validate
                    print(f"best value (so far)!")

        # make sure self.W reflects best results
        self.W = results['W']

        # add results to self
        if add_to_self:
            if not hasattr(self, 'cross_validation_results'):
                # add if there are no results in self yet
                self.cross_validation_results = results
                print("added 'cross_validation_results' to self")
            elif results['R2']['validate'] > self.cross_validation_results['R2']['validate']:
                # if not, update self if current results are better than previous results
                self.cross_validation_results = results

        # save results to pickle
        if pickle:
            pd.Series(results).to_pickle(
            f".cross_validate_{round(results['R2']['validate'],3)}.pkl"
            )

        # output
        if ouput: return results
