import sklearn
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import inv
from sklearn.cluster import KMeans
from numpy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances

def mult_precision(i1, i2, comp_label, true_label):
    # multiplicity precision
    out = min(len(comp_label[i1] & comp_label[i2]), len(true_label[i1] & true_label[i2]))/float(len(comp_label[i1] & comp_label[i2]))
    print(out)
    return out

def mult_recall(i1, i2, comp_label, true_label):
    # multiplicity recall
    out = min(len(comp_label[i1] & comp_label[i2]), len(true_label[i1] & true_label[i2]))/float(len(true_label[i1] & true_label[i2]))
    print(out)
    return out

def BCubed_precision(comp_label, true_label):
    # extended BCubed precision
    ss = np.shape(comp_label)
    ss = ss[0]

    temp_list = []
    

    
    out = np.mean([np.mean([mult_precision(i1, i2, comp_label, true_label) for i2 in range(ss) if comp_label[i1] & comp_label[i2]]) for i1 in range(ss)])
    
    return out

def BCubed_recall(comp_label, true_label):
    # extended BCubed recall
    ss = np.shape(comp_label)
    ss = ss[0]

    
    
    out = np.mean([np.mean([mult_recall(i1, i2, comp_label, true_label) for i2 in range(ss) if true_label[i1] & true_label[i2]]) for i1 in range(ss)])
    
    return out

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


'''
comp_set = [{1,2}, {2}]
true_set = [{2,3}, {1,2}]

comp_label = np.array([[1,1],[0,1]])
true_label = np.array([[0,1,1],[1,1,0]])

                  
print(soft_BCubed_recall(comp_label, true_label), BCubed_recall(comp_set, true_set))

'''


n = 11
nums = np.linspace(0,1,n)
true_label = np.array([[1,0],[1,0]])


res_p = np.zeros(n)
res_r = np.zeros(n)
res_F1 = np.zeros(n)

for i in range(n):
    comp_label = np.array([[1,0], [nums[i],1 - nums[i]]])
    res_p[i] = soft_BCubed_precision(comp_label, true_label)
    res_r[i] = soft_BCubed_recall(comp_label, true_label)
    res_F1[i] = F1_score(res_p[i], res_r[i])
        

plt.plot(nums, res_p, '-o', label = 'Soft BCubed Precision')
plt.plot(nums, res_r,  '-o', label = 'Soft BCubed Recall')
plt.plot(nums, res_F1,  '-o', label = 'Soft BCubed F-Score')
plt.xlabel('n')
plt.legend(loc='best')
plt.axis([0, 1, 0, 1.2])
plt.show()

