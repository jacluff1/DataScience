# import dependencies
import numpy as np
import matplotlib.pyplot as plt

# simulate data
N = 100
x = np.linspace(0,10,N)
y = 3.6512 + 0.98473*x + 4.21098*x**2 + np.random.randn(N)*10

# prep the Data
PHI = np.vstack( (np.ones(N),x,x**2) ).T

# fit the model
w = np.linalg.solve(PHI.T.dot(PHI), PHI.T.dot(y))
y_hat = PHI.dot(w)
R2 = 1 - np.sum( (y-y_hat)**2 ) / np.sum( (y-y.mean())**2 )
print("R-squared: {:0.3f}".format(R2))

plt.figure()
plt.scatter(x,y)
plt.plot(x,y_hat,color='red',linewidth=2)
plt.show()
