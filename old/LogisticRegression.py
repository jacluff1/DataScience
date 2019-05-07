# import external libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import internal libraries
from DataScience.Base import Base

#===============================================================================
# functions outside of class
#===============================================================================

def one_hot_encode(y):
    N = len(y)
    K = len(set(y))

    Y = np.zeros((N,K))

    for i in range(N):
        Y[i,y[i]] = 1

    return Y

def softmax(H):
    eH = np.exp(H)
    return eH / eH.sum(axis=1, keepdims=True)

def cross_entropy(Y,P_hat):
    return -np.sum(Y*np.log(P_hat))

def accuracy(y,P_hat):
    return np.mean(y == P_hat.argmax(axis=1))

# does this method actually work?
def confusion_matrix(Y,Y_hat):
    # Y = pd.Series(Y.argmax(axis=1), name='actual')
    # Y_hat = pd.Series(Y_hat.argmax(axis=1), name='predicted')
    # return pd.crosstab(Y,Y_hat)
    return np.matmul(Y.T,Y_hat)

def precision(CM):
    # TP / (TP + FP)
    return CM[1,1] / (CM[1,1] + CM[0,1])

def recall(CM):
    # TP / (TP + FN)
    return CM[1,1] / (CM[1,1] + CM[1,0])

def F_score(precission,recall):
    return 2*precission*recall / (precission+recall)

# ROC_AUC

#===============================================================================
# class
#===============================================================================

class LogisticRegression(Base):

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
        eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3

        # set up gradient descent
        epochs = int(epochs)
        J = np.zeros(epochs)

        # start with random W
        if self.K > 1:
            self.W = np.random.randn(self.P*self.K).reshape([self.P,self.K])
        else:
            self.W = np.random.randn(self.P)

        # run gradient descent
        for epoch in range(epochs):
            H = self.predict(PHI)
            P_hat = softmax(H)
            J[epoch] = (cross_entropy(Y,P_hat) + lambda1*np.sum(np.abs(W)) + lambda2*np.sum(W**2)/2)/N
            self.W -= eta*(PHI.T.dot(P_hat-Y) + lambda1*np.sign(W) + lambda2*W)/N

        return J

    # add more numerical methods as we learn them

    def solve(self,PHI,Y,**kwargs):

        # kwargs
        output = kwargs['output'] if 'output' in kwargs else False
        method = kwargs['method'] if 'method' in kwargs else "gradient descent"
        save_curve = kwargs['save_curve'] if 'save_curve' in kwargs else False

        # select solving method
        if method == "gradient_descent":
            J = self.__solve_gradient_descent(PHI,Y,**kwargs)

        # plot
        if save_curve:
            plt.figure()
            plt.plot(J)
            plt.title(f"$\\lambda_1$: {lambda1}, $\\lambda_2$: {lambda2}, $\\eta$: {eta}, epochs: {epochs}")
            filename = f"J_eta{eta}_epochs{epochs}_lambda1{lambda1}_lambda2{lambda2}.pdf"
            plt.savefig(filename)
            print(f"\nsaved {filename}")
            plt.close()

        # output
        if output: return self.W

    #===========================================================================
    # cross validation
    #===========================================================================

    # how to factor in threshold in cross validation?
    def cv_train_validate_test(self,L1,L2,T,**kwargs):

        # make sure train, validate, and test groups have been set up
        if not hasattr(self,'train'): self.__cv_tvt_add_sets(**kwargs)

        # kwargs
        output = kwargs['output'] if 'output' in kwargs else False
        pickle = kwargs['pickle'] if 'pickle' in kwargs else False
        add_to_self = kwargs['add_to_self'] if 'add_to_self' in kwargs else True
        eata = kwargs['eta'] if 'eta' in kwargs else 1e-3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
        epochs = int(epochs)
        if self.K > 1: T = [.5]

        # set up dictionary to collect results
        results = dict(eta=eta, epochs=epochs)
        bestAc_validate = 0

        # cast a net and find best outcome
        for l1 in L1:
            for l2 in L2:
                for t in T:

                    # print iteration to track progress
                    print(f"\nworking on {l1},{l2},{t}...")

                    # update kwargs
                    kwargs['lambda1'] = l1
                    kwargs['lambda2'] = l2

                    # train
                    self.solve(self.train['PHI'],self.train['Y'],**kwargs)
                    H_train = self.predict(self.train['PHI'])
                    P_hat_train = softmax(H_train)
                    ac_train = accuracy(train['Y'],P_hat_train)

                    # validate
                    H_validate = self.predict(self.validate['PHI'])
                    P_hat_validate = softmax(H_validate)
                    ac_validate = accuracy(validate['Y'],P_hat_validate)

                    # print iteration results
                    print(f"train accuracy: {ac_train}, validate accuracy {ac_validate}")

                    # update best values
                    if ac_validate > bestAc_validate:

                        # run the test set
                        H_test = self.predict(test['PHI'])
                        P_hat_test = softmax(H_test)
                        ac_test = accuracy(test['PHI'],P_hat_test)

                        # update the best values
                        results['lambda1'] = lambda1
                        results['lambda2'] = lambda2
                        if self.K > 1: results['threshold'] = t
                        results['accuracy'] = dict(train=ac_train, validate=ac_validate, test=ac_test)
                        results['W'] = self.W.copy()
                        bestAc_validate = ac_validate
                        print(f"best value (so far)!")

        # make sure self.W reflects best results
        self.W = results['W']

        # add results to self
        if add_to_self:
            if not hasattr(self, 'cross_validation_results'):
                # add if there are no results in self yet
                self.cross_validation_results = results
                print("added 'cross_validation_results' to self")
            elif results['accuracy']['validate'] > self.cross_validation_results['accuracy']['validate']:
                # if not, update self if current results are better than previous results
                self.cross_validation_results = results

        # save results to pickle
        if pickle:
            pd.Series(results).to_pickle(
            f".cross_validate_{round(results['accuracy']['validate'],3)}.pkl"
            )

        # output
        if ouput: return results
