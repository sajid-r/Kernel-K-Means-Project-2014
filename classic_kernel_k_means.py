import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
from sklearn import metrics

import os, glob

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from time import time
import numpy as np
prev = np.zeros(7095)

class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means
    
    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="polynomial", gamma=.0097, degree=2, coef0=3,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose
        
    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        '''computes the model by calculating centroids for each cluster'''

        n_samples = X.shape[0]

        K = self._get_kernel(X)
  
        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_  = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))

        self.within_distances_ = np.zeros(self.n_clusters)

        for it in xrange(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_, update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change 
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print "Converged at iteration", it + 1
                break

        self.X_fit_ = X

        prev = self.labels_
        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the 
        kernel trick."""

        sw = self.sample_weight_

        for j in xrange(self.n_clusters):
            mask = self.labels_ == j
            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom
            if update_within:
                KK = K[mask][:, mask] 
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom     #calculating distance of each point from centroid of cluster j by finding 
                                                                                #diff. b/w centroid of cluster j & similarity of it with points in cluster j

    def predict(self, X):
        '''Uses the model calculated to predict for each document the closest cluster it belongs to'''
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,update_within=False)

        return dist.argmin(axis=1)

def main():

    true_k = 4
    labels = []

    training_set = []

    path = os.getcwd()+'/classicdocs/classic/'

    for file in glob.glob(os.path.join(path, '*')):

      data = ""        
      for line in open(file) :
        data += line

      training_set.append(data)

      if 'cacm' in str(file):
          labels.append(0)
      elif 'cisi' in str(file):
          labels.append(1)
      elif 'cran' in str(file):
          labels.append(2)
      elif 'med' in str(file):
          labels.append(3)


    n_components = 20

    print 'Total Samples',len(training_set)

    print("Extracting features from the training dataset using a sparse vectorizer")
    
    # Perform an IDF normalization on the output of HashingVectorizer
    '''It turns a collection of text documents into a scipy.sparse matrix holding token occurrence counts
    This text vectorizer implementation uses the hashing trick to find the token string name to feature integer index mapping.'''
    
    hasher = HashingVectorizer(stop_words='english', non_negative=True,norm=None, binary=False)
    
    '''Transform a count matrix to a normalized tf-idf representation. It provides IDF weighting.'''        
    vectorizer = make_pipeline(hasher, TfidfTransformer(norm='l2', smooth_idf=True, sublinear_tf=False, use_idf=True))
        
    X = vectorizer.fit_transform(training_set)

    if n_components:
        print("Performing dimensionality reduction using SVD")
        '''This transformer performs linear dimensionality reduction by means of singular value decomposition (SVD)'''
        svd = TruncatedSVD(n_components)
        lsa = make_pipeline(svd, Normalizer(copy=False))
        X = lsa.fit_transform(X)

    km = KernelKMeans(n_clusters= 5, max_iter=100, verbose=1)

    km.fit_predict(X)
    predict = km.predict(X)

    print 'Adjusted_Rand_Score',metrics.adjusted_rand_score(labels, predict)
    print 'Mutual Info',metrics.adjusted_mutual_info_score(labels, predict)  
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, predict))
if __name__ == '__main__':
    main()
