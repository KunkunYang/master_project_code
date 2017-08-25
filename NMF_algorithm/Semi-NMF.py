import json
import sklearn
import random
import numpy as np
import skfuzzy as fuzz

from collections import Counter
from gensim import corpora, models
from collections import defaultdict
from numpy.linalg import inv
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances


def semi_NMF(X, numofcluster, numofiter, tol):
    
    ss = np.shape(X)
    ss = ss[0]
    
    # do K means
    kmeans = KMeans(n_clusters=numofcluster, init = 'random', n_init = 1,max_iter= 500, tol = 10**(-6)).fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # initialize
    X = np.transpose(X)

    G = np.zeros((ss,numofcluster))

    for i in range(ss):
        G[i, labels[i]] = 1


    # iteration
    last_G = G + 0.2
    last_F = []
    last_res = float("inf")



    for i in range(numofiter):

        # update F
        GTG = np.matmul(np.transpose(last_G), last_G)
        GTG_inv = inv(GTG)
        XG = np.matmul(X, last_G)
        temp_F = np.matmul(XG, GTG_inv)
        temp_F[np.isnan(temp_F)] = 0
        
        # update G
        XTF = np.matmul(np.transpose(X), temp_F)
        FTF = np.matmul(np.transpose(temp_F), temp_F)
    
        mat_1_pos = pospart(XTF)
        mat_1_neg = negpart(XTF)
        mat_2_pos = np.matmul(last_G, pospart(FTF))
        mat_2_neg = np.matmul(last_G, negpart(FTF))

        # avoid division by zero
        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)
        
        div = np.divide(mat_1_pos + mat_2_neg, mat_1_neg + mat_2_pos)
        div[np.isnan(div)] = 0
        
        temp_G = last_G*np.sqrt(div)
        temp_G[np.isnan(temp_G)] = 0
        
        # calculate residual
        a = (X - np.matmul(temp_F, np.transpose(temp_G)))**2
        res = np.sqrt(a.sum())
            
        if abs(res - last_res) < tol:
            print('Semi End: ', i-1, last_res)
            return last_F, last_G, last_res
        else:
            last_G = temp_G
            last_F = temp_F
            last_res = res

    print('Semi End: ', i, last_res)
    
    return last_F, last_G , last_res
