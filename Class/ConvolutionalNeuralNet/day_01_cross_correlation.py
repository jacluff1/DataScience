import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy
from scipy import signal

# load image
X = plt.imread("saturn.jpg")

def make_grayscale(plot=True):
    I = np.mean(X, axis=2)
    if plot:
        plt.imshow(I, cmap='gray')
        plt.savefig("saturn_gray.jpg")
        plt.close()
    return I
I = make_grayscale(plot=False)

def cross_correlation(F,plot=True,savename="something.jpg"):

    # rows,cols = I.shape[0],I.shape[1]
    # size = F.shape[0]
    # correlation = np.zeros( (rows - size , cols-size) )
    # for i in range(rows-size):
    #     for j in range(cols-size):
    #         correlation[i,j] = (I[ i:i+size , j:j+size ] * F).sum()

    # correlation = scipy.signal.correlate(I,F, mode='valid')



    if plot:
        plt.imshow(correlation, cmap='gray')
        plt.savefig(savename)
        plt.close()
    return correlation

def gaussian_filter(size):

    def gaussian(x,mean,var):
        return (1/np.sqrt(2*np.pi*var)) * np.exp(-(x-mean)**2 / (2*var))

    x = np.arange(size)

    F = np.zeros((size,size))
    mean = size//2
    var = size//3.3
    for i in range(size):
        for j in range(size):
            x_ij = np.sqrt(i**2 + j**2)
            F[i,j] = gaussian(x_ij,mean,var)

    return F

def edge_detector_filter():
    return np.array([[1,1,1],[0,0,0],[-1,-1,-1]])

# scipy.signal.correlate(mode='valid')
