# import external dependencies
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pdb
# import internal dependencies
from DataScience.Base import Base

class ArtificalNeuralNet(Base):

    def __init__(self,activation_functions,Ms):
        """
        activation_functions:
            dict of activation class instances with int keys that correspond to the layer in which they are applied. 1 <= key <= L
        Ms:
            dict of int values for the length of each layer. the keys correspond to the layer in which they are applied. 1 <= key <= L-1
        """
        # check that the length of activation_functions == length of Ms + 1
        assert len(activation_functions) == len(Ms) + 1, "There must be 1 more activation function than Ms for hidden layers"

        # it seems easier to follow counting numbers rather than whole numbers, make sure the activating function dict and Ms dict follow this convention
        assert 0 not in activation_functions, "please don't include a zero index in the activation functions\n1 <= key <= L please."
        assert 0 not in Ms, "please don't include a zero index in the node number dictionary\n1 <= key <= L-1 please."

        # assign activation_functions
        self.af = activation_functions

        # assign the number of nodes in each layer
        self.M = Ms

        # assign number of layers
        self.L = len(activation_functions)

        # assign hidden
        self.l = range(2,self.L)

    #===========================================================================
    # predict - seems to work
    #===========================================================================

    def feed_forward(self,X):

        # assign empty dict for output
        Z = {}

        # start at input layer
        Z[1] = self.af[1].f( np.matmul(X,self.W[1]) + self.b[1] )

        # go through each hidden layer + output layer
        for l in range(2,self.L+1):
            Z[l] = self.af[l].f( np.matmul(Z[l-1],self.W[l]) + self.b[l] )

        # assign Z
        self.Z = Z

    #===========================================================================
    # solve - seems to work
    #===========================================================================

    def solve_vanilla(self,*args,**kwargs):

        print("\nsolve using vanilla gradient descent...")

        #=======================================================================
        # setup
        #=======================================================================

        self.__assign_random_weights_and_biases(*args,**kwargs)

        # kwargs
        save_plot = kwargs['save_plot'] if 'save_plot' in kwargs else True
        lambda1 = kwargs['lambda1'] if 'lambda1' in kwargs else 0
        lambda2 = kwargs['lambda2'] if 'lambda2' in kwargs else 0
        eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else self.train.N
        # variables from kwargs
        batches = self.train.N//batch_size
        epochs = int(epochs)

        # set up dZ,H,dH,dW,db
        dZ,H,dH,dW,db = {},{},{},{},{}

        # set up cost function
        J_train = np.zeros(epochs*batches)
        J_validate = np.zeros_like(J_train)

        #=======================================================================
        # vanilla - gradient descent
        #=======================================================================

        for epoch in tqdm(range(epochs)):
            X,Y = self._shuffle(self.train['PHI'],self.train['Y'])
            for batch in tqdm(range(batches)):

                # get the batch data
                X_b = X[(batch*batch_size):(batch+1)*batch_size]
                Y_b = Y[(batch*batch_size):(batch+1)*batch_size]

                # feed forward
                self.feed_forward(X_b)

                # start with output layer
                dH[self.L] = self.Z[self.L] - Y_b
                dW[self.L] = (np.matmul( self.Z[self.L-1].T , dH[self.L] ) + lambda2*self.W[self.L]) / self.train.N
                db[self.L] = dH[self.L].sum(axis=0) / self.train.N
                self.W[self.L] -= eta*dW[self.L]
                self.b[self.L] -= eta*db[self.L]

                # now work back through each layer till input layer
                for l in np.arange(2,self.L)[::-1]:
                    dZ[l] = np.matmul( dH[l+1] , self.W[l+1].T )
                    dH[l] = dZ[l] * self.af[l].df(self.Z[l])
                    dW[l] = (np.matmul( self.Z[l-1].T , dH[l] ) + lambda2*self.W[l] ) / self.train.N
                    db[l] = dH[l].sum(axis=0) / self.train.N
                    self.W[l] -= eta*dW[l]
                    self.b[l] -= eta*db[l]

                # end with input layer
                dZ[1] = np.matmul( dH[2] , self.W[2].T )
                dH[1] = dZ[1] * self.af[1].df(self.Z[1])
                dW[1] = (np.matmul( X_b.T , dH[1] ) + lambda2*self.W[1] ) / self.train.N
                db[1] = dH[1].sum(axis=0)
                self.W[1] -= eta*dW[1]
                self.b[1] -= eta*db[1]

                # get training batch objective function
                index = batch + (epoch*batches)
                self.feed_forward(self.train['PHI'])
                J_train[index] = (self._cross_entropy(self.train.Y) + lambda2*self.__L2() ) / self.train.N

                # get validation batch objective function
                self.feed_forward(self.validate['PHI'])
                J_validate[index] = (self._cross_entropy(self.validate.Y) + lambda2*self.__L2() ) / self.validate.N

        # collect results
        results = {
            'J_train' : J_train,
            'J_validate' : J_validate,
            'lambda1' : lambda1,
            'lambda2' : lambda2,
            'eta' : eta,
            'epochs' : epochs,
            'batch_size' : batch_size,
            'W' : self.W,
            'b' : self.b
            }
        self.vanilla_results = pd.Series(results)

        if save_plot: self.plot_objective_functions(*args,**kwargs)

    def solve_momentum(self,*args,**kwargs):

        print("\nsolve using Nesterov momenum...")

        #=======================================================================
        # setup
        #=======================================================================

        self.__assign_random_weights_and_biases(*args,**kwargs)

        # kwargs
        save_plot = kwargs['save_plot'] if 'save_plot' in kwargs else True
        lambda1 = kwargs['lambda1'] if 'lambda1' in kwargs else 0
        lambda2 = kwargs['lambda2'] if 'lambda2' in kwargs else 0
        mu = kwargs['mu'] if 'mu' in kwargs else 0.9
        eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else self.train.N
        # variables from kwargs
        batches = self.train.N//batch_size
        epochs = int(epochs)

        # set up dZ,H,dH,dW,db
        dZ,H,dH,dW,db = {},{},{},{},{}
        vW,vb = {},{}
        for l in range(self.L+1):
            vW[l] = 0
            vb[l] = 0

        # set up cost function
        J_train = np.zeros(epochs*batches)
        J_validate = np.zeros_like(J_train)

        #=======================================================================
        # Nesterov momentum - gradient descent
        #=======================================================================

        for epoch in tqdm(range(epochs)):
            X,Y = self._shuffle(self.train['PHI'],self.train['Y'])
            for batch in tqdm(range(batches)):

                # get the batch data
                X_b = X[(batch*batch_size):(batch+1)*batch_size]
                Y_b = Y[(batch*batch_size):(batch+1)*batch_size]

                # feed forward
                self.feed_forward(X_b)

                # start with output layer
                # vanilla
                dH[self.L] = self.Z[self.L] - Y_b
                dW[self.L] = np.matmul( self.Z[self.L-1].T , dH[self.L] )
                db[self.L] = dH[self.L].sum(axis=0)
                # Nestorov momentum
                vW[self.L] = mu*vW[self.L] - eta*dW[self.L]
                vb[self.L] = mu*vb[self.L] - eta*db[self.L]
                # update
                self.W[self.L] += mu*vW[self.L] - eta*dW[self.L]
                self.b[self.L] += mu*vb[self.L] - eta*db[self.L]

                # now work back through each layer till input layer
                for l in np.arange(2,self.L)[::-1]:
                    # vanilla
                    dZ[l] = np.matmul( dH[l+1] , self.W[l+1].T )
                    dH[l] = dZ[l] * self.af[l].df(self.Z[l])
                    dW[l] = np.matmul( self.Z[l-1].T , dH[l] )
                    db[l] = dH[l].sum(axis=0)
                    # Nesterov momentum
                    vW[l] = mu*vW[l] - eta*dW[l]
                    vb[l] = mu*vb[l] - eta*db[l]
                    # update
                    self.W[l] += mu*vW[l] - eta*dW[l]
                    self.b[l] += mu*vb[l] - eta*db[l]

                # end with input layer
                # vanilla
                dZ[1] = np.matmul( dH[2] , self.W[2].T )
                dH[1] = dZ[1] * self.af[1].df(self.Z[1])
                dW[1] = np.matmul( X_b.T , dH[1] )
                db[1] = dH[1].sum(axis=0)
                # Nesterov momentum
                vW[1] = mu*vW[1] - eta*dW[1]
                vb[1] = mu*vb[1] - eta*db[1]
                # update
                self.W[1] += mu*vW[1] - eta*dW[1]
                self.b[1] += mu*vb[1] - eta*db[1]

                # get training batch objective function
                index = batch + (epoch*batches)
                self.feed_forward(self.train['PHI'])
                J_train[index] = self._cross_entropy(self.train.Y) / self.train.N

                # get validation batch objective function
                self.feed_forward(self.validate['PHI'])
                J_validate[index] = self._cross_entropy(self.validate.Y) / self.validate.N

        # collect results
        results = {
            'J_train' : J_train,
            'J_validate' : J_validate,
            'lambda1' : lambda1,
            'lambda2' : lambda2,
            'mu': mu,
            'eta' : eta,
            'epochs' : epochs,
            'batch_size' : batch_size,
            'W' : self.W,
            'b' : self.b
            }
        self.momentum_results = pd.Series(results)

        if save_plot:
            kwargs['method'] = 'momentum'
            self.plot_objective_functions(*args,**kwargs)

    def solve_RMSProp(self,*args,**kwargs):

        print("\nsolving using RMSProp with Nesterov momenumtum...")

        #=======================================================================
        # setup
        #=======================================================================

        self.__assign_random_weights_and_biases(*args,**kwargs)

        # kwargs
        save_plot = kwargs['save_plot'] if 'save_plot' in kwargs else True
        lambda1 = kwargs['lambda1'] if 'lambda1' in kwargs else 0
        lambda2 = kwargs['lambda2'] if 'lambda2' in kwargs else 0
        mu = kwargs['mu'] if 'mu' in kwargs else 0.9
        gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.999
        epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 1e-8
        eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else self.train.N
        # variables from kwargs
        batches = self.train.N//batch_size
        epochs = int(epochs)

        # set up dZ,H,dH,dW,db
        dZ,H,dH,dW,db = {},{},{},{},{}
        vW,vb,GW,Gb = {},{},{},{}
        for l in range(self.L+1):
            vW[l] = 0
            vb[l] = 0
            GW[l] = 0
            Gb[l] = 0

        # set up cost function
        J_train = np.zeros(epochs*batches)
        J_validate = np.zeros_like(J_train)

        #=======================================================================
        # RMSProp with Nesterov momentum - gradient descent
        #=======================================================================

        for epoch in tqdm(range(epochs)):
            X,Y = self._shuffle(self.train['PHI'],self.train['Y'])
            for batch in tqdm(range(batches)):

                # get the batch data
                X_b = X[(batch*batch_size):(batch+1)*batch_size]
                Y_b = Y[(batch*batch_size):(batch+1)*batch_size]

                # feed forward
                self.feed_forward(X_b)

                # start with output layer
                # vanilla
                dH[self.L] = self.Z[self.L] - Y_b
                dW[self.L] = (np.matmul( self.Z[self.L-1].T , dH[self.L] ) + lambda2*self.W[self.L] ) / self.train.N
                db[self.L] = dH[self.L].sum(axis=0) / self.train.N
                # RMSProp with Nestorov momentum
                GW[self.L] = gamma*GW[self.L] + (1-gamma)*dW[self.L]**2
                Gb[self.L] = gamma*Gb[self.L] + (1-gamma)*db[self.L]**2
                vW[self.L] = mu*vW[self.L] - (eta/np.sqrt(GW[self.L] + epsilon))*dW[self.L]
                vb[self.L] = mu*vb[self.L] - (eta/np.sqrt(Gb[self.L] + epsilon))*db[self.L]
                # update
                self.W[self.L] += vW[self.L]
                self.b[self.L] += vb[self.L]

                # now work back through each layer till input layer
                for l in np.arange(2,self.L)[::-1]:
                    # vanilla
                    dZ[l] = np.matmul( dH[l+1] , self.W[l+1].T )
                    dH[l] = dZ[l] * self.af[l].df(self.Z[l])
                    dW[l] = (np.matmul( self.Z[l-1].T , dH[l] ) + lambda2*self.W[l]) / self.train.N
                    db[l] = dH[l].sum(axis=0) / self.train.N
                    # RMSProp with Nesterov momentum
                    GW[l] = gamma*GW[l] + (1-gamma)*dW[l]**2
                    Gb[l] = gamma*Gb[l] + (1-gamma)*db[l]**2
                    vW[l] = mu*vW[l] - (eta/np.sqrt(GW[l] + epsilon))*dW[l]
                    vb[l] = mu*vb[l] - (eta/np.sqrt(Gb[l] + epsilon))*db[l]
                    # update
                    self.W[l] += vW[l]
                    self.b[l] += vb[l]

                # end with input layer
                # vanilla
                dZ[1] = np.matmul( dH[2] , self.W[2].T )
                dH[1] = dZ[1] * self.af[1].df(self.Z[1])
                dW[1] = (np.matmul( X_b.T , dH[1] ) + lambda2*self.W[1]) / self.train.N
                db[1] = dH[1].sum(axis=0) / self.train.N
                # RMSProp with Nesterov momentum
                GW[1] = gamma*GW[1] + (1-gamma)*dW[1]**2
                Gb[1] = gamma*Gb[1] + (1-gamma)*db[1]**2
                vW[1] = mu*vW[1] - (eta/np.sqrt(GW[1] + epsilon))*dW[1]
                vb[1] = mu*vb[1] - (eta/np.sqrt(Gb[1] + epsilon))*db[1]
                # update
                self.W[1] += vW[1]
                self.b[1] += vb[1]

                # get training batch objective function
                index = batch + (epoch*batches)
                self.feed_forward(self.train['PHI'])
                J_train[index] = (self._cross_entropy(self.train.Y) + lambda2*self.__L2()) / self.train.N

                # get validation batch objective function
                self.feed_forward(self.validate['PHI'])
                J_validate[index] = (self._cross_entropy(self.validate.Y) + lambda2*self.__L2() ) / self.validate.N

        # collect results
        results = {
            'J_train' : J_train,
            'J_validate' : J_validate,
            'lambda1' : lambda1,
            'lambda2' : lambda2,
            'mu': mu,
            'gamma' : gamma,
            'epsilon' : epsilon,
            'eta' : eta,
            'epochs' : epochs,
            'batch_size' : batch_size,
            'W' : self.W,
            'b' : self.b
            }
        self.RMSProp_results = pd.Series(results)

        if save_plot:
            kwargs['method'] = 'RMSProp'
            self.plot_objective_functions(*args,**kwargs)

    # not working with batch?
    def solve_Adam(self,*args,**kwargs):

        print("\nsolving using Adam...")

        #=======================================================================
        # setup
        #=======================================================================

        self.__assign_random_weights_and_biases(*args,**kwargs)

        # kwargs
        save_plot = kwargs['save_plot'] if 'save_plot' in kwargs else True
        lambda1 = kwargs['lambda1'] if 'lambda1' in kwargs else 0
        lambda2 = kwargs['lambda2'] if 'lambda2' in kwargs else 0
        mu = kwargs['mu'] if 'mu' in kwargs else 0.9
        gamma = kwargs['gamma'] if 'gamma' in kwargs else 0.999
        epsilon = kwargs['epsilon'] if 'epsilon' in kwargs else 1e-8
        eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else self.train.N
        # variables from kwargs
        batches = self.train.N//batch_size
        epochs = int(epochs)

        # set up dZ,H,dH,dW,db
        dZ,H,dH,dW,db = {},{},{},{},{}
        mW,mb,vW,vb = {},{},{},{}
        for l in range(self.L+1):
            mW[l] = 0
            mb[l] = 0
            vW[l] = 0
            vb[l] = 0

        # set up cost function
        J_train = np.zeros(epochs*batches+1)
        J_validate = np.zeros_like(J_train)

        #=======================================================================
        # RMSProp with Nesterov momentum - gradient descent
        #=======================================================================

        for epoch in tqdm(range(1,epochs)):
            X,Y = self._shuffle(self.train['PHI'],self.train['Y'])
            for batch in tqdm(range(batches)):

                # get the batch data
                X_b = X[(batch*batch_size):(batch+1)*batch_size]
                Y_b = Y[(batch*batch_size):(batch+1)*batch_size]

                # feed forward
                self.feed_forward(X_b)

                # start with output layer
                # vanilla
                dH[self.L] = self.Z[self.L] - Y_b
                dW[self.L] = np.matmul( self.Z[self.L-1].T , dH[self.L] )
                db[self.L] = dH[self.L].sum(axis=0)
                # adam
                mW[self.L] = mu*mW[self.L] + (1-mu)*dW[self.L]
                mb[self.L] = mu*mb[self.L] + (1-mu)*db[self.L]
                vW[self.L] = mu*vW[self.L] + (1-gamma)*dW[self.L]**2
                vb[self.L] = mu*vb[self.L] + (1-gamma)*db[self.L]**2
                # update
                self.W[self.L] -= eta/np.sqrt(vW[self.L]/(1-gamma**epoch) + epsilon) * mW[self.L]/(1-mu**epoch)
                self.b[self.L] -= eta/np.sqrt(vb[self.L]/(1-gamma**epoch) + epsilon) * mb[self.L]/(1-mu**epoch)

                # now work back through each layer till input layer
                for l in np.arange(2,self.L)[::-1]:
                    # vanilla
                    dZ[l] = np.matmul( dH[l+1] , self.W[l+1].T )
                    dH[l] = dZ[l] * self.af[l].df(self.Z[l])
                    dW[l] = np.matmul( self.Z[l-1].T , dH[l] )
                    db[l] = dH[l].sum(axis=0)
                    # adam
                    mW[l] = mu*mW[l] + (1-mu)*dW[l]
                    mb[l] = mu*mb[l] + (1-mu)*db[l]
                    vW[l] = mu*vW[l] + (1-gamma)*dW[l]**2
                    vb[l] = mu*vb[l] + (1-gamma)*db[l]**2
                    # update
                    self.W[l] -= eta/np.sqrt(vW[l]/(1-gamma**epoch) + epsilon) * mW[l]/(1-mu**epoch)
                    self.b[l] -= eta/np.sqrt(vb[l]/(1-gamma**epoch) + epsilon) * mb[l]/(1-mu**epoch)

                # end with input layer
                # vanilla
                dZ[1] = np.matmul( dH[2] , self.W[2].T )
                dH[1] = dZ[1] * self.af[1].df(self.Z[1])
                dW[1] = np.matmul( X_b.T , dH[1] )
                db[1] = dH[1].sum(axis=0)
                # adam
                mW[1] = mu*mW[1] + (1-mu)*dW[1]
                mb[1] = mu*mb[1] + (1-mu)*db[1]
                vW[1] = mu*vW[1] + (1-gamma)*dW[1]**2
                vb[1] = mu*vb[1] + (1-gamma)*db[1]**2
                # update
                self.W[1] -= eta/np.sqrt(vW[1]/(1-gamma**epoch) + epsilon) * mW[1]/(1-mu**epoch)
                self.b[1] += eta/np.sqrt(vb[1]/(1-gamma**epoch) + epsilon) * mb[1]/(1-mu**epoch)

                # get training batch objective function
                index = batch + (epoch*batches)
                self.feed_forward(self.train['PHI'])
                J_train[index] = self._cross_entropy(self.train.Y) / self.train.N

                # get validation batch objective function
                self.feed_forward(self.validate['PHI'])
                J_validate[index] = self._cross_entropy(self.validate.Y) / self.validate.N

        # collect results
        results = {
            'J_train' : J_train[1:],
            'J_validate' : J_validate[1:],
            'lambda1' : lambda1,
            'lambda2' : lambda2,
            'mu': mu,
            'gamma' : gamma,
            'epsilon' : epsilon,
            'eta' : eta,
            'epochs' : epochs,
            'batch_size' : batch_size,
            'W' : self.W,
            'b' : self.b
            }
        self.adam_results = pd.Series(results)

        if save_plot:
            kwargs['method'] = 'adam'
            self.plot_objective_functions(*args,**kwargs)

    def plot_objective_functions(self,*args,**kwargs):

        # kwargs
        method = kwargs['method'] if 'method' in kwargs else 'vanilla'
        plot_validate = kwargs['plot_validate'] if 'plot_validate' in kwargs else False

        savename = "J/J"

        # plot
        if not os.path.isdir("J"): os.mkdir("J")
        fig,ax = plt.subplots()

        if 'vanilla' in method:
            if not hasattr(self,'vanilla_results'):
                self.solve_vanilla(*args,**kwargs)
            ax.plot(self.vanilla_results.J_train, label='Vanilla - Training')
            if plot_validate:
                ax.plot(self.vanilla_results.J_validate, label='Vanilla - Validation')
            savename += '_vanilla'
            lambda1 = self.vanilla_results.lambda1
            lambda2 = self.vanilla_results.lambda2
            eta = self.vanilla_results.eta

        if 'momentum' in method:
            if not hasattr(self,'momentum_results'):
                self.solve_momentum(*args,**kwargs)
            ax.plot(self.momentum_results.J_train, label='Momentum - Training')
            if plot_validate:
                ax.plot(self.momentum_results.J_validate, label='Momentum - Validation')
            savename += '_momentum'
            lambda1 = self.momentum_results.lambda1
            lambda2 = self.momentum_results.lambda2
            eta = self.momentum_results.eta

        if 'RMSProp' in method:
            if not hasattr(self,'RMSProp_results'):
                self.solve_RMSProp(*args,**kwargs)
            ax.plot(self.RMSProp_results.J_train, label='RMSProp - Training')
            if plot_validate:
                ax.plot(self.RMSProp_results.J_validate, label="RMSProp - Validation")
            savename += '_RMSProp'
            lambda1 = self.RMSProp_results.lambda1
            lambda2 = self.RMSProp_results.lambda2
            eta = self.RMSProp_results.eta

        if 'adam' in method:
            if not hasattr(self,'adam_results'):
                self.solve_Adam(*args,**kwargs)
            ax.plot(self.adam_results.J_train, label='Adam - Training')
            if plot_validate:
                ax.plot(self.adam_results.J_validate, label='Adam - Validation')
            savename += '_adam'
            lambda1 = self.adam_results.lambda1
            lambda2 = self.adam_results.lambda2
            eta = self.adam_results.eta

        ax.set_xlabel("batch + (epochs x baches)")
        ax.set_ylabel("J")
        fig.suptitle(f"$\\eta$: {eta}, $\\lambda_1$: {lambda1}, $\\lambda_2$: {lambda2}")
        ax.legend(loc='best')
        savename += ".pdf"
        fig.savefig(savename)
        plt.close(fig)
        print(f"saved {savename}")

    #===========================================================================
    # cross validation - need to check
    #===========================================================================

    def train_validate_test(self,*args,**kwargs):

        # kwargs
        output = kwargs['output'] if 'output' in kwargs else True
        pickle = kwargs['pickle'] if 'pickle' in kwargs else True
        assign = kwargs['assign'] if 'assign' in kwargs else True
        method = kwargs['method'] if 'method' in kwargs else 'vanilla'

        # # if the desired results are not assigned already, get the vanilla results
        # if hasattr(self,f"{results}_results"):
        #     print(f"\nusing {results}_results")
        #     solve_results = getattr(self,f"{results}_results")
        # else:
        #     print(f"{results} not found. Looking for vanilla results...")
        #     if not hasattr(self,f"vanilla_results"):
        #         print("\nvanilla results not found\ngetting vanilla results...")
        #         self.solve_vanilla(*args,**kwargs)
        #     solve_results = self.vanilla_results
        if 'results' in kwargs:
            assert hasattr(self,kwargs['results']), "Results not found! try either using a solve method or load a set of results."
            solve_results = getattr(self,f"{results}_results")
        else:
            if method == 'vanilla':
                self.solve_vanilla(*args,**kwargs)
                solve_results = self.vanilla_results
            elif method == 'momentum':
                self.solve_momentum(*args,**kwargs)
                solve_results = self.momentum_results
            elif method == 'RMSProp':
                self.solve_RMSProp(*args,**kwargs)
                solve_results = self.RMSProp_results
            elif method == 'adam':
                self.solve_Adam(*args,**kwargs)
                solve_results = self.adam_results
            else:
                raise KeyError("I'm sorry, these are not the methods you are looking for...")

        # set weights and bias
        self.W = solve_results.W
        self.b = solve_results.b

        accuracy = {}

        # train
        self.feed_forward(self.train['PHI'])
        accuracy['train'] = self._accuracy(self.train['Y'],self.Z[self.L])

        # validate
        Z_validate = self.feed_forward(self.validate['PHI'])
        accuracy['validate'] = self._accuracy(self.validate['Y'],self.Z[self.L])

        # test
        self.feed_forward(self.test['PHI'])
        accuracy['test'] = self._accuracy(self.test['Y'],self.Z[self.L])
        P_hat = self.Z[self.L]
        y_hat = P_hat.argmax(axis=1)

        # collect results
        results = {
            'accuracy'  :   accuracy,
            'P_hat'     :   P_hat
            }

        try:
            Y_hat = self._one_hot_encode(y_hat)
            CM = np.array( np.matmul(self.test.Y.T,Y_hat), dtype=np.int32 )
            results['Y_hat'] = Y_hat
            results['CM'] = CM
        except:
            print("some goram problem with confusion matrix!")

        results.update(solve_results)

        # assign if current results are the best
        if assign:
            if not hasattr(self,'cross_validation_results'):
                self.cross_validation_results = results
            elif accuracy['validate'] > self.cross_validation_results['accuracy']['validate']:
                self.cross_validation_results = results

        # pickle
        if pickle:

            # find a list of previously pickled best results
            DIR = np.array(os.listdir())
            filter = ['results_' in x for x in DIR]
            DIR = DIR[filter]

            # if there are past results, save current results only if they are better than any previous results
            if len(DIR) > 0:
                best = np.array([ x[ x.find("_")+1 : x.find(".pkl") ] for x in DIR ], dtype=np.float32).max()
                if accuracy['validate'] > best:
                    print("\nfound new best result!")
                    pd.Series(results).to_pickle("results_{:0.4}.pkl".format(accuracy['validate']))
                else:
                    print("\nno such luck...")
            # if there are no results, just save the current results
            else:
                print("\nfound new best result!")
                pd.Series(results).to_pickle("results_{:0.4}.pkl".format(accuracy['validate']))

        print("Accuracy from this round: {:0.4f}".format(accuracy['validate']))
        # pd.Series(results).to_pickle("results_{:0.4f}.pkl".format(accuracy['test']))

        # output
        if output: return results

    #===========================================================================
    # helper functions - seems to work
    #===========================================================================

    def __assign_random_weights_and_biases(self,*args,**kwargs):

        print("\nassigning random weights and biases...")

        # kwargs
        seed = kwargs['seed'] if 'seed' in kwargs else 0
        np.random.seed(seed)

        self.__check_for_data_sets(*args,**kwargs)

        # assign empty dictionaries for weights and biases
        W,b = {},{}

        # add weights and biases for input layer
        W[1] = np.random.randn(self.train['PHI'].shape[1],self.M[1])
        b[1] = np.random.randn(self.M[1])

        # add weights and biases for hidden layers:
        for l in self.l:
            W[l] = np.random.randn(self.M[l-1],self.M[l])
            b[l] = np.random.randn(self.M[l])

        # add weights and biases for ouptu layer
        W[self.L] = np.random.randn(self.M[self.L-1],self.train['Y'].shape[1])
        b[self.L] = np.random.randn(self.train['Y'].shape[1])

        # assign weights and biases
        self.W = W
        self.b = b
        # print("\nadded 'W' and 'b' as attributes")

    def __check_for_data_sets(self,*args,**kwargs):

        print("\nchecking for data...")

        # check for training data
        if not hasattr(self,'train'):

            assert len(args) == 2, "\ndata sets not detected\nno arguments provided to get them\nplease either add datasets manually or provide PHI,Y for entire set\nplease use method self._cv_tvt_data_sets(X,Y),\nOR, just put X,Y into the arguments"

            print("\ndata sets not detected\ngeting data sets...")
            self._cv_tvt_data_sets(*args,**kwargs)

    def __L2(self):
        L2 = 0
        for W in self.W.values():
            L2 += (W**2).sum()
        return L2/2
