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


def mult_precision(i1, i2, comp_label, true_label):
    # multiplicity precision
    out = min(len(comp_label[i1] & comp_label[i2]), len(true_label[i1] & true_label[i2]))/float(len(comp_label[i1] & comp_label[i2]))
    return out

def mult_recall(i1, i2, comp_label, true_label):
    # multiplicity recall
    out = min(len(comp_label[i1] & comp_label[i2]), len(true_label[i1] & true_label[i2]))/float(len(true_label[i1] & true_label[i2]))
    return out

def BCubed_precision(comp_label, true_label):
    # extended BCubed precision
    ss = np.shape(comp_label)
    ss = ss[0]

    out = np.mean([np.mean([mult_precision(i1, i2, comp_label, true_label) for i2 in range(ss) if comp_label[i1] & comp_label[i2]]) for i1 in range(ss)])
    return out

def BCubed_recall(comp_label, true_label):
    # extended BCubed recall
    ss = np.shape(comp_label)
    ss = ss[0]

    out = np.mean([np.mean([mult_recall(i1, i2, comp_label, true_label) for i2 in range(ss) if true_label[i1] & true_label[i2]]) for i1 in range(ss)])
    
    return out


def F1_score(precision, recall):
    
    return 2.0*(precision*recall)/(precision+recall)


# BCubed_F_Score = F1_score(BCubed_precision(), BCubed_recall())
