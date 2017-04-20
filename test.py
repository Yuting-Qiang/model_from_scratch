import models
from models import gen_data
import numpy as np

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
