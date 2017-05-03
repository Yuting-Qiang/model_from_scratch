import numpy as np

class LinearRegression:

    def __init__(self):
        self.D = 0
        self.N = 0

    def fit_normal_equation(self, features, targets, add_intercept = True):

        self.N = features.shape[0]
        self.D = features.shape[1]
        self.weights = np.zeros(self.D)

        if add_intercept:
            self.add_intercept = True
            intercept = np.ones((self.N, 1))
            features = np.hstack((intercept, features))
            self.D += 1
            self.weights = np.hstack(([0], self.weights))

        features_trans = np.matrix(np.transpose(features))
        multi_fea_fea_t = np.matrix(np.dot(features_trans, features))
        inver_multi_feat = np.matrix(multi_fea_fea_t.I)

        self.weights = np.dot(inver_multi_feat, np.dot(features_trans, targets.reshape(targets.size, 1)))

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









