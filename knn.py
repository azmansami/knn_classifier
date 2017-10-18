#/usr/bin/env python

#k nearest neighbourhood classifier
import numpy as np
import scipy.spatial.distance as spd
from collections import Counter

class knn:
    def __init__(self):
        pass

    def train(self, x, y):
        #x is a NxD matrix, where N is the no of image and D is unfolded length of
        #each image.
        #y is the lable of each image. y is same length as x (N)
        self.X_train=x
        self.y_train=y

    def predict(self,x,k=1):
        """
        Predicts classes of each datapoint/image

        Input: x is a NxD matrix, where N is the no of datapoints/image to be tested and D is unfolded
        length of each datapoint/image
        k: k nearest neighbour, default 1
        """
        dist=self.compute_distances_no_loops(x)
        return self.predict_labels(dist,k)

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: X, test datapoint/image
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #computing L2 euclidean distance between test and train datapoint/image
        dists=spd.cdist(X,self.X_train,'euclidean')
        return dists

    def predict_labels(self, dists, k):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
        gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #finding idx of k smallest distance points
            t=(np.argsort(dists[i,:]))[:k]
            closest_y=self.y_train[t]
            vote_res=Counter(closest_y).most_common()
            if len(vote_res)==k:
                y_pred[i]=np.min(closest_y)
            else:
                y_pred[i]=int(vote_res[0][0])
        return y_pred
