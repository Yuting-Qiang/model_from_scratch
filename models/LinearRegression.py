import numpy as np

class LinearRegression:

    def __init__(self):
        self.D = 0
        self.N = 0

    def fit(self, features, targets, learning_rate=1e-5, num_iter = 300000, add_intercept=False):
        #fit a linear regression model using gradient decent
        self.N = features.shape[0]
        self.D = features.shape[1]
        self.weights = np.zeros(self.D)

        if add_intercept:
            self.add_intercept = True
            intercept = np.ones((self.N, 1))
            features = np.hstack((intercept, features))
            self.weights = np.append([0], self.weights)

        for i in range(1, num_iter):
            predictions = np.dot(features, self.weights)
            errors = predictions - targets
            gradients = np.dot(features.T, errors)

            self.weights -= learning_rate*gradients/self.N

    def predict(self, x):
        if self.add_intercept == True:
            x = np.hstack((np.array([1]), x))

        return np.dot(x, self.weights)









