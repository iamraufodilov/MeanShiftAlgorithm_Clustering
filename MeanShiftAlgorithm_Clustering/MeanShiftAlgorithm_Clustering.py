# import libraries
import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs

# creadte data
centers = [[3,3,3],[4,5,5],[3,10,10]]
X, _ = make_blobs(n_samples=700, centers = centers, cluster_std=0.5)

# create model
MSh = MeanShift()

# train model
MSh.fit(X)
labels = MSh.labels_
cluster_centers = MSh.cluster_centers_
print(cluster_centers)
