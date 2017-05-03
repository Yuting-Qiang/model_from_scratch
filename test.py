'''
to do list:
    compare my model with sklearn
    compare logistic regression with Gaussian NB

'''
import models
from models import gen_data
import numpy as np

'''
classification
'''
'''
features, targets = gen_data.gaussian_data(5000)
clf = models.LogisticRegression()
clf.fit(features, targets, add_intercept=True)

t_features, t_targets = gen_data.gaussian_data(1000)

pred_targets = np.zeros(t_targets.shape)

for i in range(2000):
    pred_targets[i] = clf.predict(t_features[i])

print(pred_targets)
print ("accuracy is %f" % (len(np.where( pred_targets[:] == t_targets)[0]) \
                           /2000.0))
'''

'''regression'''
weights = np.array([1.5, 2])
features, targets = gen_data.linear_data(weights, 1000, d=2)
clf = models.LinearRegression()
clf.fit_normal_equation(features, targets, add_intercept=True)

t_features, t_targets = gen_data.linear_data(weights, 200, d=2)

pred_targets = np.zeros(t_targets.shape)

for i in range(200):
    pred_targets[i] = clf.predict(t_features[i])

n = pred_targets.size
pairs = np.hstack((pred_targets.reshape(n, 1), t_targets.reshape(n, 1)))
print pairs
print("loss is %f" % (np.mean(np.square(pred_targets - t_targets))/ \
      np.mean(t_targets)))

