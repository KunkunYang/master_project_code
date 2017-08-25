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

# comp_label: labels returned by clustering algorithm
# true_label: true labels or human annotated labels
# numlabel: total number of labels

def Purity(comp_label, true_label, numlabel):
    
    # comp_label must start from 0!
    
    c = numlabel
    ss = np.shape(true_label)
    ss = ss[0]
    
    # assign to each cluster
    clusters = []
    
    for i in range(c):
        temp = []
        for n in range(ss):
            if i in comp_label[n]:
                temp.append(n)

        clusters.append(temp)

    # count labels in each cluster
    count_label = []

    # create true label dictionary
    a = []
    for i in range(ss):
        a = a + true_label[i]

    a = set(a)
    labels = list(a)

    dict_init = {}
    for i in range(np.size(labels)):
        dict_init[labels[i]] = 0
        
    for i in range(c):
        
        temp = dict_init.fromkeys(dict_init, 0)
        
        for j in clusters[i]:
            for k in true_label[j]:
                temp[k] = temp[k] + 1


        count_label.append(temp)
    
    Purity = 0

    for i in range(c):
        max_n = count_label[i][max(count_label[i], key=count_label[i].get)]
        #tot_n = sum(count_label[i].values())

        Purity = Purity + (1.0/ss)*max_n

    return Purity


def InvPurity(comp_label, true_label, numlabel):

    
    # comp_label must start from 0!
    
    c = numlabel
    ss = np.shape(true_label)
    ss = ss[0]
    
    # assign to each cluster
    clusters = []
    
    for i in range(c):
        temp = []
        for n in range(ss):
            if i in comp_label[n]:
                temp.append(n)

        clusters.append(temp)

    # count labels in each cluster
    count_label = []

    # create true label dictionary
    a = []
    for i in range(ss):
        a = a + true_label[i]

    a = set(a)
    labels = list(a)

    dict_init = {}
    for i in range(np.size(labels)):
        dict_init[labels[i]] = 0
    
    for i in range(c):
        tem = dict_init.fromkeys(dict_init, 0)
        
        for j in clusters[i]:
            for k in true_label[j]:
                tem[k] = tem[k] + 1


        count_label.append(tem)

    
    InvPurity = 0

    for i in labels:
        max_n = count_label[np.argmax([count_label[j][i] for j in range(c)])][i]
        #tot_n = sum([count_label[j][i] for j in range(c)])

        InvPurity = InvPurity + (1.0/ss)*max_n

    return InvPurity

def F_purity(comp_label, true_label, numlabel):

    
    # comp_label must start from 0!
    
    c = numlabel
    
    ss = np.shape(true_label)
    ss = ss[0]
    
    # assign to each cluster
    clusters = []
    
    for i in range(c):
        temp = []
        for n in range(ss):
            if i in comp_label[n]:
                temp.append(n)

        clusters.append(temp)

    # count labels in each cluster
    count_label = []

    # create true label dictionary
    a = []
    for i in range(ss):
        a = a + true_label[i]

    a = set(a)
    labels = list(a)

    dict_init = {}
    for i in range(np.size(labels)):
        dict_init[labels[i]] = 0
    
    for i in range(c):
        tem = dict_init.fromkeys(dict_init, 0)
        
        for j in clusters[i]:
            for k in true_label[j]:
                tem[k] = tem[k] + 1

        count_label.append(tem)

    # count total number of elements in each cluster and each label
    count_c = []
    
    for i in range(c):
        count_c.append(sum(count_label[i].values()))
        
    count_l = {}
    for i in labels:
        count_l[i] = sum([count_label[j][i] for j in range(c)])

    # F-purity
    F = 0

    # avoid division by zero
    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)
    
    for i in labels:
        temp = []
        
        for j in range(c):
            r_ij = np.divide(count_label[j][i], count_c[j])
            
            if np.isnan(r_ij):
                r_ij= 0
                
            p_ij = np.divide(count_label[j][i], count_l[i])
            
            if np.isnan(p_ij):
                p_ij= 0
                
            F_ij = np.divide((2*r_ij*p_ij), (r_ij + p_ij))

            if np.isnan(F_ij):
                F_ij= 0
            
            temp.append(F_ij)

        F_j = max(temp)
        F = F + (count_l[i]/ss)*F_j

    return F
