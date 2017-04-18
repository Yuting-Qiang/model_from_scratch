import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def gaussian_data(num_observations):
    np.random.seed(1)

    x1 = np.random.multivariate_normal([0, 0], [[1, 0.75], [0.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

    simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
    simulated_separableish_labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)));

    fig = plt.figure(figsize=(12,8))
    plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1], c = simulated_separableish_labels, alpha=4)
    fig.savefig('temp.png')

    return simulated_separableish_features, simulated_separableish_labels

