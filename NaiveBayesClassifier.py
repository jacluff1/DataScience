# import dependencies
import numpy as np
import pdb

#==============================================================================#
# functions
#==============================================================================#

def Gaussian(x,mu,var):
    return 1/np.sqrt(2*np.pi*var) * np.exp( -(x-mu)**2 / (2*var) )

def Multinomial(x):
    pass

def Bernoulli(x):
    pass

#===============================================================================
# class definition
#===============================================================================

class NaiveBayes:

    def __init__(self, *liklihood_args, **options_kwargs):

        # add all input as attributes
        self.update_liklihood_and_options(*liklihood_args, **options_kwargs)

    def update_liklihood_and_options(self, *liklihood_args, **options_kwargs):

        # make sure the liklihood argmuments contain at least the priors and some other liklihood argument
        assert len(liklihood_args) >= 2, "liklihood arguments must contain at least priors and one argument to go in the liklyhood function"

        # convert all the liklihood arguments to numpy arrays
        args = []
        for arg in liklihood_args:
            args.append(np.array(arg))

        # save the priors and the liklihood specific args
        self.priors = args[0]
        self.liklihood_args = args[1:]

        # set up default options: Gaussian liklihood
        options = {
            'liklihood'     : Gaussian,
            'log'           : True,
            'return_phat'   : False,
            'print_output'  : False
        }

        # update options with options_kwargs
        options.update(options_kwargs)
        self.options = options

    def predict(self,x):

        # # get the number of classes
        # K = len(self.liklihood_args)

        # find the posteriors
        if self.options['log']:
            phat = np.log( self.options['liklihood'](x, *self.liklihood_args) ) + np.log( self.priors )
        else:
            phat = self.options['liklihood'](x, *self.liklihood_args) * self.priors

        # find the prediction
        yhat = np.argmax(phat)

        if self.options['print_output']:
            # find the string for the function
            s = str(self.options['liklihood'])
            s = s[ s.find('function')+9 : s.rfind('at')-1 ]
            print(f"\n{s} liklihood, yhat: {yhat}, phat: {phat}")

        if self.options['return_phat']:
            return phat
        else:
            return yhat

#===============================================================================
# example
#===============================================================================

def examples():
    # priors
    p_f = 0.5149
    p_m = 1 - p_f
    priors = np.array([p_m, p_f])

    mu = np.array([70, 65])
    sigma = np.array([4, 3.5])
    var = sigma**2

    example1 = NaiveBayes(priors, mu, var, print_output=True)
    example1.predict(66)

    example2 = NaiveBayes(priors, print_output=True, liklihood='Multinomial')
    example2.predict(66)

    example3 = NaiveBayes(priors, print_output=True, liklihood='Bernoulli')
    example3.predict(66)
