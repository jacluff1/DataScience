import numpy as np
import pdb
from DataScience.ANN import ArtificalNeuralNet,one_hot_encode
import DataScience.ActivationFunctions as AF

# get simulated data dimentions
D = 2
K = 3
N = int(K*1e3)

# simulate input data
X0 = np.random.randn((N//K),D) + np.array([2,2])
X1 = np.random.randn((N//K),D) + np.array([0,-2])
X2 = np.random.randn((N//K),D) + np.array([-2,2])
X = np.vstack((X0, X1, X2))

# simulate target data
y = np.array([0]*(N//K) + [1]*(N//K) + [2]*(N//K))
Y = one_hot_encode(y)

# set up ANN
M = {1:8, 2:8, 3:8, 4:8, 5:8}
af = {1:AF.ReLU(), 2:AF.ReLU(), 3:AF.ReLU(), 4:AF.ReLU(), 5:AF.ReLU(), 6:AF.softmax()}

# make instance of ANN
ann = ArtificalNeuralNet(X,Y,af,M)

# solve
ann.back_propagation(X,Y)
