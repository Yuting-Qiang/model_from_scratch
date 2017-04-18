'''
This code is an implementation of Gaussian Naive Bayes
@author Yuting Qiang
@data 2017/04/18
'''

import numpy as np
import math
'''
Gaussian Naive Bayes
to do list:
    multi-classification
    discrete features
note:
    variance is independent of y
'''
class GaussianNB:

    def __init__(self):
        self.classes = 2

    def sigmoid(score):
        return 1.0/1.0+exp(score)

    def fit(self, features, targets):

        num_features = features.shape[0]
        d = features.shape[1]

        self.weights = np.zeros(d)
        mu_array = np.zeros([d, self.classes])
        sigma_array = np.zeros(d)


        for i in range(0, d):

            #calculate mean and standard deviation
            mu_array[i, 0] = np.mean(features[np.where(targets[:] \
                                                       == 0), i])
            mu_array[i, 1] = np.mean(features[np.where(targets[:] \
                                                       == 1), i])

            sigma_array[i] = np.std(features[:, i])

            self.weights[i] = (mu_array[i, 0] - mu_array[i, 1]) / sigma_array[i] \
                **2

            pi = len(np.where(targets[:] == 0)[0])/float(num_features)

            print("pi = %f" % (pi))
            self.weights0 = math.log(pi/(1-pi)) + np.sum((mu_array[:, 1]**2 - \
                                                     mu_array[:, 0]**2) \
                                                     / (2*(sigma_array[:]**2)))

            print(self.weights)
            print ("weights0 = %f" % (self.weights0))


    def predict(self, feature):
        res = np.dot(feature, self.weights)+self.weights0
        if res > 0.5:
            return 0
        else:
            return 1
