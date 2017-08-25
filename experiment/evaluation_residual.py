import json
import sklearn
import random
import numpy as np
import skfuzzy as fuzz
import math

from collections import Counter
from gensim import corpora, models
from collections import defaultdict
from numpy.linalg import inv
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances


def abs_secondderivative(x, num):
    ss = np.size(x)
    out = np.zeros(ss)

    for i in range(ss):
        if i != 0 and i!= ss-1:
            out[i] = (x[i+1] + x[i-1] - 2 * x[i])/(num[i] - num[i-1])**2
        out[0] = None
        out[ss-1] = None
    return abs(out)

def Silhouette(X, comp_label, numlabel):

    
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

            if np.size(cluster[int(lab)]) == 0:
                bi = float("inf")
            else:
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
                
        sil[i] = (b - a) / (max(a, b))

    return np.mean(sil)



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

def F1_score(precision, recall):
    
    return 2.0*(precision*recall)/(precision+recall)

def pospart(mat):
    out = 0.5*(abs(mat) + mat)
    return out

def negpart(mat):
    out = 0.5*(abs(mat) - mat)
    return out

def semi_NMF(X, numofcluster, numofiter, tol):
    
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


    # iteration
    last_G = G + 0.2
    last_F = []
    last_res = float("inf")

    #import ipdb; ipdb.set_trace()


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

# name and embedding of data
file = input("Dataset name (amazon, yelp, frames, artificial): ")
embedding = input("Embedding name (glove, fasttext, count, (mult for artificial data)): ")

# load data
data = np.load('pre-processed_dataset/data+label_'+file + '_' +embedding+'.npz')

data_matrix = data['data_matrix']
labels =  data['labels']

data_matrix = data_matrix[0:50]
labels = labels[0:50]

# size
data_size = (np.shape(data_matrix))[0]
emb_size = (np.shape(data_matrix))[1]

# set and list of true label
s_true_label = [set(labels[i]) for i in range(data_size)]
l_true_label = labels


# get all labels
list_all = []
for i in labels:
    list_all = list_all + i


unique_labels = list(np.unique(list_all))
labels_size = np.size(unique_labels)

num_labels_list = [5,8,11,14,17,20]

list_res_semi =np.zeros(np.size(num_labels_list))
list_res_conv =np.zeros(np.size(num_labels_list))
list_res_fuzz2 =np.zeros(np.size(num_labels_list))
list_res_fuzz11 =np.zeros(np.size(num_labels_list))



for j in range(np.size(num_labels_list)):
    ####### fuzzy k-means with m = 2
    numofiter = 500
    tol = 10**(-6)
        
    MM2 = fuzz.cluster.cmeans(np.transpose(data_matrix), num_labels_list[j], 2, error = tol, maxiter=numofiter, init = None)
    G_fuzz2 = np.transpose(MM2[1])
    res_fuzz2 = (MM2[4])[-1]
    list_res_fuzz2[j] = res_fuzz2

    ####### fuzzy k-means with m = 1.1
    numofiter = 500
    tol = 10**(-6)
        
    MM11 = fuzz.cluster.cmeans(np.transpose(data_matrix), num_labels_list[j], 1.1, error = tol, maxiter=numofiter, init = None)
    G_fuzz11 = np.transpose(MM11[1])
    res_fuzz11 = (MM11[4])[-1]
    list_res_fuzz11[j] = res_fuzz11
    
        
    ####### NMF
    numofiter = 500
    tol = 10**(-6)
    
    F, G_semi, res_semi = semi_NMF(data_matrix, num_labels_list[j], numofiter, tol)
    W, G_conv, res_conv = convex_NMF(data_matrix, num_labels_list[j], numofiter, tol)

    list_res_semi[j] = res_semi
    list_res_conv[j] = res_conv
                    

# absolute second order derivative
sd_semi = abs_secondderivative(list_res_semi, num_labels_list)
sd_conv = abs_secondderivative(list_res_conv, num_labels_list)
sd_fuzz2 = abs_secondderivative(list_res_fuzz2, num_labels_list)
sd_fuzz11 = abs_secondderivative(list_res_fuzz11, num_labels_list)


np.savez('residual_'+file + '_' +embedding+'.npz',
         list_res_semi = list_res_semi, list_res_conv = list_res_conv, list_res_fuzz2 = list_res_fuzz2, list_res_fuzz11 = list_res_fuzz11)
