import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances


def imbalance_impact_knn(data, label):
    pos_num = sum(label == 1)
    neg_num = sum(label == -1)
    pos_idx = np.nonzero(label == 1)
    neg_idx = np.nonzero(label == -1)
    pos_data = data[pos_idx]
    rr = neg_num / pos_num
    k = 5

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(data)
    distances, knn_idx = nbrs.kneighbors(pos_data)

    p2 = np.zeros(pos_num)
    p2old = np.zeros(pos_num)
    knn_idx = np.delete(knn_idx, 0, 1)
    for i in range(pos_num):
        p2[i] = np.intersect1d(knn_idx[i], neg_idx).size / k
        p2old[i] = p2[i]
        if p2[i] == 1:
            dist = pairwise_distances(pos_data[i].reshape(1, -1), data).reshape(-1)
            sort_idx = np.argsort(dist)
            nearest_pos = np.nonzero(label[sort_idx] == 1)[0][1]
            p2[i] = (nearest_pos - 1) / nearest_pos
    p1 = 1 - p2
    # px = p2 * rr * p1 / (p2 + rr * p1)
    # px = p2 * (rr * p1 / (p2 + rr * p1) - p1)
    px = (rr * p1 / (p2 + rr * p1) - p1)

    pm = np.mean(px)

    return px, pm


data_name = sys.argv[1]
load_data = np.load(data_name)
data = load_data['data']
label = np.ravel(load_data['label'])
pos_num = sum(label == 1)
neg_num = sum(label == -1)
print('The imbalance ratio is %.4f' % (neg_num / pos_num))

ibi3, bi3 = imbalance_impact_knn(data, label)
print('The bi3 value is %.4f' % bi3)


