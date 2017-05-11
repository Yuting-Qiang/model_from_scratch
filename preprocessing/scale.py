import numpy as np

def scale(data):
    # scale data to 0~1 for data[:,0], data[:, 1], ...

    mins = np.amin(data, axis = 0)
    maxs = np.amax(data, axis = 0)
    mins_maxs = maxs - mins

    data = (data-mins)/mins_maxs.astype(np.float32)

    return data

