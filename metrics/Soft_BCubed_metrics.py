import sklearn
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import inv
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances

def soft_mult_precision(i1, i2, comp_label, true_label):
    # multiplicity precision
    out = min(np.dot(comp_label[i1], comp_label[i2]), np.dot(true_label[i1], true_label[i2]))/float(np.dot(comp_label[i1], comp_label[i2]))

        
    return out

def soft_mult_recall(i1, i2, comp_label, true_label):
    # multiplicity recall
    out = min(np.dot(comp_label[i1], comp_label[i2]), np.dot(true_label[i1], true_label[i2]))/float(np.dot(true_label[i1], true_label[i2]))
        
    return out

def soft_BCubed_precision(comp_label, true_label):
    # extended BCubed precision
    ss = np.shape(comp_label)
    ss = ss[0]

    tot_list = []
    
    for i1 in range(ss):
        temp_list = []
        
        for i2 in range(ss):
            if np.dot(comp_label[i1], comp_label[i2]) > 0.0:
                temp_list.append(soft_mult_precision(i1, i2, comp_label, true_label))

        if temp_list != []:
            tot_list.append(np.mean(temp_list))
            
    if tot_list == []:
        out = 0.0
    else:
        out = np.mean(np.array(tot_list))

    return out
    

def soft_BCubed_recall(comp_label, true_label):
    # extended BCubed recall
    ss = np.shape(comp_label)
    ss = ss[0]
    
    tot_list = []
    
    for i1 in range(ss):
        temp_list = []
        
        for i2 in range(ss):
            if np.dot(true_label[i1], true_label[i2]) > 0.0:
                temp_list.append(soft_mult_recall(i1, i2, comp_label, true_label))

        if temp_list != []:
            tot_list.append(np.mean(temp_list))

    if tot_list == []:
        out = 0.0
    else:
        out = np.mean(np.array(tot_list))
        
    return out

def F1_score(precision, recall):
    
    return 2.0*(precision*recall)/(precision+recall)



# soft_BCubed_F_Score = F1_score(soft_BCubed_precision(), soft_BCubed_recall())
