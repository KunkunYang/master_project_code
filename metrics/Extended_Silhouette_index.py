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

def Silhouette(X, comp_label, numlabel):

    # X: data matrix
    # comp_label: labels returned by clustering algorithm
    # numlabel: total number of labels
    
    c = numlabel
    
    ss = np.shape(comp_label)
    ss = ss[0]

    cluster = [[] for _ in range(c)]

    # assign each sentence to cluster
    
    for i in range(ss):
        for j in range(np.size(comp_label[i])):
            cluster[int(comp_label[i][j])].append(i)
            
    
    # compute sil for each sentence:
    sil = np.zeros(ss)

    all_clus = np.linspace(0,c-1,c)
    all_clus = all_clus.astype(int)

    distances = euclidean_distances(X)
    
    
    for i in range(ss):
        not_in_clus = [x for x in all_clus if x not in comp_label[i]]
         
        dist_a = 0
        dist_b = []
                       
        size_in_clus = 0
                       
        for lab in comp_label[i]:
            size_in_clus = size_in_clus + np.size(cluster[int(lab)]) - 1
                       
            for elem in cluster[int(lab)]:
                if elem != i:
                    dist_a = dist_a + distances[i, elem]

        for lab in not_in_clus:
            temp = 0
            
            for elem in cluster[int(lab)]:
                temp = temp + distances[i, elem]

            if np.size(cluster[int(lab)]) != 0:
                bi = temp / np.size(cluster[int(lab)])
                dist_b.append(bi)


        if size_in_clus == 0:
            a = 0
        else:
            a = dist_a / size_in_clus

        if dist_b == []:
            b = 0
        else:
            b = min(dist_b)

        if max(a, b) == 0:
            sil[i] = 0
        else:
            sil[i] = (b - a) / (max(a, b))


    return np.mean(sil)
