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


def convex_NMF(X, numofcluster, numofiter, tol):
    
    ss = np.shape(X)
    ss = ss[0]
    
    # do K means
    kmeans = KMeans(n_clusters=numofcluster, init = 'random', n_init = 1, max_iter= 500, tol = 10**(-6)).fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # initialize
    X = np.transpose(X)

    G = np.zeros((ss,numofcluster))

    for i in range(ss):
        G[i, labels[i]] = 1

    G_0 = G + 0.2
    D = np.diag(1.0/sum(G))
    W_0 = np.dot(G_0, D)

    # iteration
    last_G = G_0
    last_W = W_0
    last_res = []

    XTX = np.dot(np.transpose(X), X)
    Xp = pospart(XTX)
    Xn = negpart(XTX)

    for i in range(numofiter):

        # update G
        GWT = np.dot(last_G, np.transpose(last_W))
        
        mat_G_1_pos = np.dot(Xp, last_W)
        mat_G_1_neg = np.dot(Xn, last_W)
        mat_G_2_pos = np.dot(np.dot(GWT, Xp), last_W)
        mat_G_2_neg = np.dot(np.dot(GWT, Xn), last_W)

        # avoid division by zero
        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)
        
        div_G = np.divide(mat_G_1_pos + mat_G_2_neg, mat_G_1_neg + mat_G_2_pos)
        div_G[np.isnan(div_G)] = 0
        
        temp_G = last_G*np.sqrt(div_G)
        temp_G[np.isnan(temp_G)] = 0
        # update W
        WGTG = np.dot(np.dot(last_W, np.transpose(temp_G)), temp_G)
        
        mat_W_1_pos = np.dot(Xp, temp_G)
        mat_W_1_neg = np.dot(Xn, temp_G)
        mat_W_2_pos = np.dot(Xp, WGTG)
        mat_W_2_neg = np.dot(Xn, WGTG)
    
        # avoid division by zero
        div_W = np.divide(mat_W_1_pos + mat_W_2_neg, mat_W_1_neg + mat_W_2_pos)
        div_W[np.isnan(div_W)] = 0
        
        temp_W = last_W*np.sqrt(div_W)
        temp_W[np.isnan(temp_W)] = 0
        
        # calculate residual
        a = (X - np.dot(np.dot(X, temp_W), np.transpose(temp_G)))**2
        res = np.sqrt(a.sum())

        #print(i, res)

        if res > last_res:
            print('Convex End: ', i-1, last_res)
            return last_W, last_G, last_res
        else:
            last_G = temp_G
            last_W = temp_W
            last_res = res

        #print(i, res)
        
    print('Convex End: ', i, last_res)

    return last_W, last_G, last_res
