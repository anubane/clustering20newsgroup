"""
model.py

This file implements the Logistic Regression model for classification.
"""

import numpy as np

class Kmeans(object):

    def __init__(self, dist = 'euc'):
        """
        Initialise the model. Here, we only initialise the weights as 'None',
        as the input size of the model will become apparent only when some
        training samples will be given to us.

        @add_bias: Whether to add bias.
        """

        # Initialise model parameters placeholders. Don't need another placeholder
        # for bias, as it can be incorporated in the weights matrix by adding
        # another feature in the input vector whose value is always 1.
        self.n_clusters = None
        self.n_data_points = None
        self.n_dimensions = None
        self.n_iterations = None

        self.centers = None
        self.labels = None
        self.dist = dist

    def distance(self, a, b):
        if self.dist == 'euc':
            return self.euc(a, b)

    def euc(self,a,b):
        ''' Input - @a, @b are two vectors
            Output - return a single value which is the difference of @a and @b
        '''
        ''' YOUR CODE HERE'''
        return np.linalg.norm((a-b), axis=1)
        #dist = (a - b) ** 2         # using broadcasting property
        #dist = np.sum(dist, axis=1)
        #return np.sqrt(dist)
        ''' YOUR CODE ENDS'''

    def cluster(self, X, k=1, n_iter=10, debug=True):
        self.n_clusters = k
        self.n_iterations = n_iter
        self.n_data_points = X.shape[0]
        self.n_dimensions = X.shape[1]

        # INITIALIZATION
        #self.centers = np.random.randn(self.n_clusters,self.n_dimensions)
        
        '''
        Pick @k cluster centers randomly from the data points (@X) and store the sampled points in a variable called @self.centers
        '''
        ''' YOUR CODE HERE'''
        self.centers = X.copy()
        np.random.shuffle(self.centers)
        self.centers = self.centers[:self.n_clusters]
        # print ("X.shape"+str(X.shape)+"self.centers"+str(self.centers.shape))
        self.labels = np.zeros((self.n_data_points,1))
        ''' YOUR CODE ENDS'''

        '''
            Update the cluster centers now.
        '''
        ''' YOUR CODE HERE'''
        # Loop here.
        for i in range(n_iter):
            # dist = np.zeros((self.n_data_points, 1))
            # Allocate label to each document
            for j in range(self.n_data_points):
                self.labels[j] = self.distance(self.centers, X[j]).argmin(axis=0)
                # self.labels[j] = np.repeat(X[j][np.newaxis, :], self.n_clusters, axis=0)
        
            # Find new cluster centers
            for l in range(self.n_clusters):
                cluster_points = np.where(self.labels==l, X, None)   # pick the points from X for which label is l else None
                cluster_points = cluster_points[cluster_points != np.array(None)]
                self.centers[l] = np.mean(cluster_points, axis=0)
        ''' YOUR CODE ENDS'''

        return self.labels
