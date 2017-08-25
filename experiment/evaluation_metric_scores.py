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
        if i % 100 == 0:
            print('Semi:', i)
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
        if i % 100 == 0:
            print('Convex:', i)
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

#data_matrix = data_matrix[0:50]
#labels = labels[0:50]

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

# average result of 5 rounds with different initialization
numofrounds = 5
# number of labels retrieved
numoflabel = 5

# store metric scores
Sil_kmeans = np.zeros(numofrounds)
Sil_semi = np.zeros((numoflabel, numofrounds))
Sil_conv =np.zeros((numoflabel, numofrounds))
Sil_fuzz2 =np.zeros((numoflabel, numofrounds))
Sil_fuzz11 =np.zeros((numoflabel, numofrounds))

BCubed_p_kmeans = np.zeros(numofrounds)
BCubed_r_kmeans = np.zeros(numofrounds)
F1_kmeans  = np.zeros(numofrounds)

BCubed_p_semi = np.zeros((numoflabel, numofrounds))
BCubed_r_semi = np.zeros((numoflabel, numofrounds))
F1_semi  = np.zeros((numoflabel, numofrounds))


BCubed_p_conv = np.zeros((numoflabel, numofrounds))
BCubed_r_conv = np.zeros((numoflabel, numofrounds))
F1_conv = np.zeros((numoflabel, numofrounds))

BCubed_p_fuzz2 = np.zeros((numoflabel, numofrounds))
BCubed_r_fuzz2 = np.zeros((numoflabel, numofrounds))
F1_fuzz2  = np.zeros((numoflabel, numofrounds))

BCubed_p_fuzz11 = np.zeros((numoflabel, numofrounds))
BCubed_r_fuzz11 = np.zeros((numoflabel, numofrounds))
F1_fuzz11  = np.zeros((numoflabel, numofrounds))

Purity_kmeans = np.zeros(numofrounds)
InvPurity_kmeans = np.zeros(numofrounds)
F_purity_kmeans = np.zeros(numofrounds)

Purity_semi= np.zeros((numoflabel, numofrounds))
InvPurity_semi = np.zeros((numoflabel, numofrounds))
F_purity_semi = np.zeros((numoflabel, numofrounds))

Purity_conv = np.zeros((numoflabel, numofrounds))
InvPurity_conv = np.zeros((numoflabel, numofrounds))
F_purity_conv = np.zeros((numoflabel, numofrounds))

Purity_fuzz2 = np.zeros((numoflabel, numofrounds))
InvPurity_fuzz2  = np.zeros((numoflabel, numofrounds))
F_purity_fuzz2 = np.zeros((numoflabel, numofrounds))

Purity_fuzz11 = np.zeros((numoflabel, numofrounds))
InvPurity_fuzz11  = np.zeros((numoflabel, numofrounds))
F_purity_fuzz11 = np.zeros((numoflabel, numofrounds))

list_res_semi =np.zeros(numofrounds)
list_res_conv =np.zeros(numofrounds)
list_res_fuzz2 =np.zeros(numofrounds)
list_res_fuzz11 =np.zeros(numofrounds)

# thresholding scores
Sil_semi_thresh = np.zeros(numofrounds)
Sil_conv_thresh = np.zeros(numofrounds)
Sil_fuzz2_thresh = np.zeros(numofrounds)
Sil_fuzz11_thresh = np.zeros(numofrounds)

BCubed_p_semi_thresh = np.zeros(numofrounds)
BCubed_r_semi_thresh = np.zeros(numofrounds)
F1_semi_thresh  = np.zeros(numofrounds)

BCubed_p_conv_thresh = np.zeros(numofrounds)
BCubed_r_conv_thresh =np.zeros(numofrounds)
F1_conv_thresh = np.zeros(numofrounds)

BCubed_p_fuzz2_thresh = np.zeros(numofrounds)
BCubed_r_fuzz2_thresh =np.zeros(numofrounds)
F1_fuzz2_thresh = np.zeros(numofrounds)

BCubed_p_fuzz11_thresh = np.zeros(numofrounds)
BCubed_r_fuzz11_thresh =np.zeros(numofrounds)
F1_fuzz11_thresh = np.zeros(numofrounds)

Purity_semi_thresh = np.zeros(numofrounds)
InvPurity_semi_thresh = np.zeros(numofrounds)
F_purity_semi_thresh = np.zeros(numofrounds)

Purity_conv_thresh = np.zeros(numofrounds)
InvPurity_conv_thresh = np.zeros(numofrounds)
F_purity_conv_thresh = np.zeros(numofrounds)

Purity_fuzz2_thresh = np.zeros(numofrounds)
InvPurity_fuzz2_thresh = np.zeros(numofrounds)
F_purity_fuzz2_thresh = np.zeros(numofrounds)

Purity_fuzz11_thresh = np.zeros(numofrounds)
InvPurity_fuzz11_thresh = np.zeros(numofrounds)
F_purity_fuzz11_thresh = np.zeros(numofrounds)

avr_lab_retrieved_semi_thresh = np.zeros(numofrounds)
avr_lab_retrieved_conv_thresh = np.zeros(numofrounds)
avr_lab_retrieved_fuzz2_thresh = np.zeros(numofrounds)
avr_lab_retrieved_fuzz11_thresh = np.zeros(numofrounds)

# Compute true Silhouette
labelsinnum = []

for i in range(data_size):
    temp = []
    for j in labels[i]:
        temp.append(unique_labels.index(j))
    labelsinnum.append(temp)
    
true_sil = Silhouette(data_matrix, labelsinnum, labels_size)

print('True Sil:', true_sil)


for i in range(numofrounds):
    print('round: ', i)
    ####### Kmeans
    kmeans = KMeans(n_clusters=labels_size, init = 'random', n_init = 1, max_iter= 500, tol = 10**(-6)).fit(data_matrix)
    kmeans_labels = kmeans.labels_

    set_kmeans_labels = []
    list_kmeans_labels = []
        
    for obj in kmeans_labels:
        set_kmeans_labels.append({obj})
        list_kmeans_labels.append([obj])

    # Sil
    Sil_kmeans[i] = Silhouette(data_matrix, list_kmeans_labels, labels_size)

    # Purity
    Purity_kmeans[i] = Purity(list_kmeans_labels, l_true_label, labels_size)
    InvPurity_kmeans[i] = InvPurity(list_kmeans_labels, l_true_label, labels_size)
    F_purity_kmeans[i] = F_purity(list_kmeans_labels, l_true_label, labels_size)
                        
               
    # BCubed
    # precision, recall, F1
    BCubed_p_kmeans[i] = BCubed_precision(set_kmeans_labels, s_true_label)
    BCubed_r_kmeans[i] = BCubed_recall(set_kmeans_labels, s_true_label)
    F1_kmeans[i] = F1_score(BCubed_p_kmeans[i], BCubed_r_kmeans[i])

    ####### fuzzy c-means with m = 1.1

    numofiter = 500
    tol = 10**(-6)
    
    MM11 = fuzz.cluster.cmeans(np.transpose(data_matrix), labels_size, 1.1, error = tol, maxiter=numofiter, init = None) 
    G_fuzz11 = np.transpose(MM11[1])
    res_fuzz11 = (MM11[4])[-1]
    list_res_fuzz11[i] = res_fuzz11

    ####### fuzzy c-means with m = 2

    numofiter = 500
    tol = 10**(-6)
    
    MM2 = fuzz.cluster.cmeans(np.transpose(data_matrix), labels_size, 2, error = tol, maxiter=numofiter, init = None)
    G_fuzz2 = np.transpose(MM2[1])
    res_fuzz2 = (MM2[4])[-1]
    list_res_fuzz2[i] = res_fuzz2
    
    ####### NMF
    numofiter = 500
    tol = 10**(-6)
    
    F, G_semi, res_semi = semi_NMF(data_matrix, labels_size, numofiter, tol)
    W, G_conv, res_conv = convex_NMF(data_matrix, labels_size, numofiter, tol)

    list_res_semi[i] = res_semi
    list_res_conv[i] = res_conv
                    
    # thresholding
    comp_label_semi = []
    comp_label_conv = []
    comp_label_fuzz2 = []
    comp_label_fuzz11 = []
    l_comp_label_semi = []
    l_comp_label_conv = []
    l_comp_label_fuzz2 = []
    l_comp_label_fuzz11 = []

    for m in range(data_size):
        comp_label_semi.append(set([pos for pos,v in enumerate(G_semi[m,:]) if v >= np.mean(G_semi[m,:])]))
        comp_label_conv.append(set([pos for pos,v in enumerate(G_conv[m,:]) if v >= np.mean(G_conv[m,:])]))
        comp_label_fuzz2.append(set([pos for pos,v in enumerate(G_fuzz2[m,:]) if v >= np.mean(G_fuzz2[m,:])]))
        comp_label_fuzz11.append(set([pos for pos,v in enumerate(G_fuzz11[m,:]) if v >= np.mean(G_fuzz11[m,:])]))
    
        l_comp_label_semi.append([pos for pos,v in enumerate(G_semi[m,:]) if v >= np.mean(G_semi[m,:])])
        l_comp_label_conv.append([pos for pos,v in enumerate(G_conv[m,:]) if v >= np.mean(G_conv[m,:])])
        l_comp_label_fuzz2.append([pos for pos,v in enumerate(G_fuzz2[m,:]) if v >= np.mean(G_fuzz2[m,:])])
        l_comp_label_fuzz11.append([pos for pos,v in enumerate(G_fuzz11[m,:]) if v >= np.mean(G_fuzz11[m,:])])

        avr_lab_retrieved_semi_thresh[i] += np.size([pos for pos,v in enumerate(G_semi[m,:]) if v >= np.mean(G_semi[m,:])])
        avr_lab_retrieved_conv_thresh[i] += np.size([pos for pos,v in enumerate(G_conv[m,:]) if v >= np.mean(G_conv[m,:])])
        avr_lab_retrieved_fuzz2_thresh[i] += np.size([pos for pos,v in enumerate(G_fuzz2[m,:]) if v >= np.mean(G_fuzz2[m,:])])
        avr_lab_retrieved_fuzz11_thresh[i] += np.size([pos for pos,v in enumerate(G_fuzz11[m,:]) if v >= np.mean(G_fuzz11[m,:])])

    avr_lab_retrieved_semi_thresh[i] = avr_lab_retrieved_semi_thresh[i]/data_size
    avr_lab_retrieved_conv_thresh[i] = avr_lab_retrieved_conv_thresh[i]/data_size
    avr_lab_retrieved_fuzz2_thresh[i] = avr_lab_retrieved_fuzz2_thresh[i]/data_size
    avr_lab_retrieved_fuzz11_thresh[i] = avr_lab_retrieved_fuzz11_thresh[i]/data_size

    # Sil
    Sil_semi_thresh[i] = Silhouette(data_matrix, l_comp_label_semi, labels_size)
    Sil_conv_thresh[i] = Silhouette(data_matrix, l_comp_label_conv, labels_size)
    Sil_fuzz2_thresh[i] = Silhouette(data_matrix, l_comp_label_fuzz2, labels_size)
    Sil_fuzz11_thresh[i] = Silhouette(data_matrix, l_comp_label_fuzz11, labels_size)
    
    # Purity
    Purity_conv_thresh[i] = Purity(l_comp_label_conv, l_true_label, labels_size)
    InvPurity_conv_thresh[i] = InvPurity(l_comp_label_conv, l_true_label, labels_size)
    F_purity_conv_thresh[i] = F_purity(l_comp_label_conv, l_true_label, labels_size)
                
                
    Purity_semi_thresh[i] = Purity(l_comp_label_semi, l_true_label, labels_size)
    InvPurity_semi_thresh[i] = InvPurity(l_comp_label_semi, l_true_label, labels_size)
    F_purity_semi_thresh[i] = F_purity(l_comp_label_semi, l_true_label, labels_size)

    Purity_fuzz2_thresh[i] = Purity(l_comp_label_fuzz2, l_true_label, labels_size)
    InvPurity_fuzz2_thresh[i] = InvPurity(l_comp_label_fuzz2, l_true_label, labels_size)
    F_purity_fuzz2_thresh[i] = F_purity(l_comp_label_fuzz2, l_true_label, labels_size)

    Purity_fuzz11_thresh[i] = Purity(l_comp_label_fuzz11, l_true_label, labels_size)
    InvPurity_fuzz11_thresh[i] = InvPurity(l_comp_label_fuzz11, l_true_label, labels_size)
    F_purity_fuzz11_thresh[i] = F_purity(l_comp_label_fuzz11, l_true_label, labels_size)
            
    # BCubed
    # precision, recall, F1
    BCubed_p_conv_thresh[i] = BCubed_precision(comp_label_conv, s_true_label)
    BCubed_r_conv_thresh[i] = BCubed_recall(comp_label_conv, s_true_label)
    F1_conv_thresh[i] = F1_score(BCubed_p_conv_thresh[i], BCubed_r_conv_thresh[i])

    BCubed_p_semi_thresh[i] = BCubed_precision(comp_label_semi, s_true_label)
    BCubed_r_semi_thresh[i] = BCubed_recall(comp_label_semi, s_true_label)
    F1_semi_thresh[i] = F1_score(BCubed_p_semi_thresh[i], BCubed_r_semi_thresh[i])

    BCubed_p_fuzz2_thresh[i] = BCubed_precision(comp_label_fuzz2, s_true_label)
    BCubed_r_fuzz2_thresh[i] = BCubed_recall(comp_label_fuzz2, s_true_label)
    F1_fuzz2_thresh[i] = F1_score(BCubed_p_fuzz2_thresh[i], BCubed_r_fuzz2_thresh[i])

    BCubed_p_fuzz11_thresh[i] = BCubed_precision(comp_label_fuzz11, s_true_label)
    BCubed_r_fuzz11_thresh[i] = BCubed_recall(comp_label_fuzz11, s_true_label)
    F1_fuzz11_thresh[i] = F1_score(BCubed_p_fuzz11_thresh[i], BCubed_r_fuzz11_thresh[i])

    
    # retrieve maximum n labels
    for j in range(numoflabel):

        print('n labels: ', j+1)
                   
        comp_label_semi = []
        comp_label_conv = []
        comp_label_fuzz2 = []
        comp_label_fuzz11 = []
        l_comp_label_semi = []
        l_comp_label_conv = []
        l_comp_label_fuzz2 = []
        l_comp_label_fuzz11 = []


        numoflabels = j+1
                   
        for m in range(data_size):
            comp_label_semi.append(set(G_semi[m,:].argsort()[-numoflabels:][::-1]))
            comp_label_conv.append(set(G_conv[m,:].argsort()[-numoflabels:][::-1]))
            comp_label_fuzz2.append(set(G_fuzz2[m,:].argsort()[-numoflabels:][::-1]))
            comp_label_fuzz11.append(set(G_fuzz11[m,:].argsort()[-numoflabels:][::-1]))
    
            l_comp_label_semi.append(G_semi[m,:].argsort()[-numoflabels:][::-1])
            l_comp_label_conv.append(G_conv[m,:].argsort()[-numoflabels:][::-1])
            l_comp_label_fuzz2.append(G_fuzz2[m,:].argsort()[-numoflabels:][::-1])
            l_comp_label_fuzz11.append(G_fuzz11[m,:].argsort()[-numoflabels:][::-1])
            
            
                
        # Silhouette index
        Sil_semi[j,i] = Silhouette(data_matrix, l_comp_label_semi, labels_size)
        Sil_conv[j,i] = Silhouette(data_matrix, l_comp_label_conv, labels_size)
        Sil_fuzz2[j,i] = Silhouette(data_matrix, l_comp_label_fuzz2, labels_size)
        Sil_fuzz11[j,i] = Silhouette(data_matrix, l_comp_label_fuzz11, labels_size)
           
        # Purity
        Purity_conv[j,i] = Purity(l_comp_label_conv, l_true_label, labels_size)
        InvPurity_conv[j,i] = InvPurity(l_comp_label_conv, l_true_label, labels_size)
        F_purity_conv[j,i] = F_purity(l_comp_label_conv, l_true_label, labels_size)
                
                
        Purity_semi[j,i] = Purity(l_comp_label_semi, l_true_label, labels_size)
        InvPurity_semi[j,i] = InvPurity(l_comp_label_semi, l_true_label, labels_size)
        F_purity_semi[j,i] = F_purity(l_comp_label_semi, l_true_label, labels_size)

        Purity_fuzz2[j,i] = Purity(l_comp_label_fuzz2, l_true_label, labels_size)
        InvPurity_fuzz2[j,i] = InvPurity(l_comp_label_fuzz2, l_true_label, labels_size)
        F_purity_fuzz2[j,i] = F_purity(l_comp_label_fuzz2, l_true_label, labels_size)

        Purity_fuzz11[j,i] = Purity(l_comp_label_fuzz11, l_true_label, labels_size)
        InvPurity_fuzz11[j,i] = InvPurity(l_comp_label_fuzz11, l_true_label, labels_size)
        F_purity_fuzz11[j,i] = F_purity(l_comp_label_fuzz11, l_true_label, labels_size)
            
        # BCubed
        # precision, recall, F1
        BCubed_p_conv[j,i] = BCubed_precision(comp_label_conv, s_true_label)
        BCubed_r_conv[j,i] = BCubed_recall(comp_label_conv, s_true_label)
        F1_conv[j,i] = F1_score(BCubed_p_conv[j,i], BCubed_r_conv[j,i])

        BCubed_p_semi[j,i] = BCubed_precision(comp_label_semi, s_true_label)
        BCubed_r_semi[j,i] = BCubed_recall(comp_label_semi, s_true_label)
        F1_semi[j,i] = F1_score(BCubed_p_semi[j,i], BCubed_r_semi[j,i])

        BCubed_p_fuzz2[j,i] = BCubed_precision(comp_label_fuzz2, s_true_label)
        BCubed_r_fuzz2[j,i] = BCubed_recall(comp_label_fuzz2, s_true_label)
        F1_fuzz2[j,i] = F1_score(BCubed_p_fuzz2[j,i], BCubed_r_fuzz2[j,i])

        BCubed_p_fuzz11[j,i] = BCubed_precision(comp_label_fuzz11, s_true_label)
        BCubed_r_fuzz11[j,i] = BCubed_recall(comp_label_fuzz11, s_true_label)
        F1_fuzz11[j,i] = F1_score(BCubed_p_fuzz11[j,i], BCubed_r_fuzz11[j,i])


        
np.savez('result_'+file + '_' +embedding+'.npz', BCubed_p_conv = BCubed_p_conv, BCubed_r_conv = BCubed_r_conv, F1_conv = F1_conv,
         BCubed_p_semi = BCubed_p_semi, BCubed_r_semi = BCubed_r_semi, F1_semi = F1_semi,
         BCubed_p_kmeans = BCubed_p_kmeans, BCubed_r_kmeans = BCubed_r_kmeans, F1_kmeans  = F1_kmeans,
         BCubed_p_fuzz2 = BCubed_p_fuzz2, BCubed_r_fuzz2 = BCubed_r_fuzz2, F1_fuzz2  = F1_fuzz2,
         BCubed_p_fuzz11 = BCubed_p_fuzz11, BCubed_r_fuzz11 = BCubed_r_fuzz11, F1_fuzz11  = F1_fuzz11,
         
         Purity_conv = Purity_conv, InvPurity_conv = InvPurity_conv, F_purity_conv = F_purity_conv,
         Purity_semi = Purity_semi, InvPurity_semi = InvPurity_semi, F_purity_semi = F_purity_semi,
         Purity_kmeans = Purity_kmeans, InvPurity_kmeans = InvPurity_kmeans, F_purity_kmeans = F_purity_kmeans,
         Purity_fuzz11 = Purity_fuzz11, InvPurity_fuzz11 = InvPurity_fuzz11, F_purity_fuzz11 = F_purity_fuzz11,
         Purity_fuzz2 = Purity_fuzz2, InvPurity_fuzz2 = InvPurity_fuzz2, F_purity_fuzz2 = F_purity_fuzz2,
         
         Sil_semi = Sil_semi, Sil_conv = Sil_conv, Sil_kmeans = Sil_kmeans,  Sil_fuzz2 = Sil_fuzz2, Sil_fuzz11 = Sil_fuzz11,
         list_res_semi = list_res_semi, list_res_conv = list_res_conv, list_res_fuzz2 = list_res_fuzz2, list_res_fuzz11 = list_res_fuzz11,

         true_sil = true_sil, avr_lab_retrieved_semi_thresh = avr_lab_retrieved_semi_thresh, avr_lab_retrieved_conv_thresh = avr_lab_retrieved_conv_thresh,
         avr_lab_retrieved_fuzz2_thresh = avr_lab_retrieved_fuzz2_thresh, avr_lab_retrieved_fuzz11_thresh = avr_lab_retrieved_fuzz11_thresh,

         Sil_semi_thresh= Sil_semi_thresh, Sil_conv_thresh= Sil_conv_thresh, Sil_fuzz2_thresh= Sil_fuzz2_thresh, Sil_fuzz11_thresh= Sil_fuzz11_thresh, 
         BCubed_p_conv_thresh = BCubed_p_conv_thresh, BCubed_r_conv_thresh = BCubed_r_conv_thresh, F1_conv_thresh = F1_conv_thresh,
         BCubed_p_semi_thresh = BCubed_p_semi_thresh, BCubed_r_semi_thresh = BCubed_r_semi_thresh, F1_semi_thresh = F1_semi_thresh,
         BCubed_p_fuzz2_thresh = BCubed_p_fuzz2_thresh, BCubed_r_fuzz2_thresh = BCubed_r_fuzz2_thresh, F1_fuzz2_thresh = F1_fuzz2_thresh,
         BCubed_p_fuzz11_thresh = BCubed_p_fuzz11_thresh, BCubed_r_fuzz11_thresh = BCubed_r_fuzz11_thresh, F1_fuzz11_thresh = F1_fuzz11_thresh,
         
         
         Purity_conv_thresh = Purity_conv_thresh, InvPurity_conv_thresh = InvPurity_conv_thresh, F_purity_conv_thresh = F_purity_conv_thresh,
         Purity_semi_thresh = Purity_semi_thresh, InvPurity_semi_thresh = InvPurity_semi_thresh, F_purity_semi_thresh = F_purity_semi_thresh,
         Purity_fuzz2_thresh = Purity_fuzz2_thresh, InvPurity_fuzz2_thresh = InvPurity_fuzz2_thresh, F_purity_fuzz2_thresh = F_purity_fuzz2_thresh,
         Purity_fuzz11_thresh = Purity_fuzz11_thresh, InvPurity_fuzz11_thresh = InvPurity_fuzz11_thresh, F_purity_fuzz11_thresh = F_purity_fuzz11_thresh)
