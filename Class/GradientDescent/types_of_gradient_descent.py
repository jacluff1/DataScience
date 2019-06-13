import numpy as np

def cross_entropy(Y,P_hat):
    return -np.sum(Y*np.log(P_hat))

def softmax(H):
    eH = np.exp(H)
    return eH / eH.sum(axis=1, keepdims=True)

def shuffle_data(X,Y):

    # shuffle mask
    shuffle = np.arange(X.shape[0])
    np.random.shuffle(shuffle)

    # output
    return X[shuffle],Y[shuffle]

def batch_gradient_descent(X,Y,batch_size,**kwargs):

    # kwargs
    eta = kwargs['eta'] if 'eta' in kwargs else 1e-3
    epochs = kwargs['epochs'] if 'epochs' in kwargs else 1e3
    epochs = int(epochs)
    batch_size = int(batch_size)

    # get dimentions
    N,D = X.shape
    if Y.shape[0] == Y.size:
        K = 1
    else:
        K = Y.shape[1]

    # set up gradient descent
    J = np.zeros(epochs)
    W = np.random.randn(D,K)

    for epoch in epochs:

        Je = 0

        # shuffle data
        X1,Y1 = shuffle_data(X,Y)

        i_start,i_end = 0,batch_size
        N_batches = X.shape[0]//batch_size

        for n in range(N_batches):

            X2,Y2 = X1[i_start:i_end],Y1[i_start:i_end]
            P_hat = softmax(X2.dot(W))
            Je += cross_entropy(Y2,P_hat)
            W -= eta*X2.T.dot(P_hat-Y2)

            i_start = i_end
            i_end += batch_size

        J[epoch] = Je
