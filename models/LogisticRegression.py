'''
@author Yuting Qiang
@data 2017/04/19
This is an implementation of LogisticRegression
referrence: Generative And Discriminative Classifiers: Naive Bayes and Logistic Regression
https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf
'''

import numpy as np

class LogisticRegression:

    def sigmoid(self, score):
        return 1.0/(1.0+np.exp(score))

    def log_likelihood(self, features, targets, weights):
        scores = np.dot(features, weights)
        ll = np.sum(targets*scores - np.log(1+np.exp(scores)))
        return ll

    def fit(self, features, targets, learning_rate= 5e-1, num_steps = 300000, add_intercept = False):

        if add_intercept:
            self.add_intercept = True
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
        else:
            self.add_intercept = False

        n = features.shape[0]
        d = features.shape[1]

        self.weights = np.zeros(d)

        for step in xrange(num_steps):

            scores = np.dot(features, self.weights)
            predictions = self.sigmoid(scores)

            output_error_signal = predictions - targets
            gradient = np.dot(features.T, output_error_signal)
            self.weights += learning_rate * gradient
            #self.weights += learning_rate*np.dot(features.T,\
            #                                                  targets-predictions)

            if step % 10000 == 0:
                print self.log_likelihood(features, targets, self.weights)
                print self.weights

        print self.weights

    def predict(self, feature):

        if self.add_intercept == True:
            feature = np.hstack((np.array([1]), feature))

        score = np.dot(feature, self.weights)
        return 1-np.round(self.sigmoid(score))


