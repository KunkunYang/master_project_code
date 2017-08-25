import json
import numpy as np
from collections import Counter
from gensim import corpora, models
from collections import defaultdict

def read_json_line(filename):

    data = []
    
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    return data

def read_data_sent_plus(dataset_name, embedding_name, a):
    
    # read and convert data from json file to matrix
    data = read_json_line('data/embed_'+ embedding_name + '_'+ a + '_data_'+ dataset_name +'.json')
    ss = np.size(data)

    dim = np.size(data[0]['embedding_' + embedding_name][0])

    # count total number of turns
    count = 0
    for i in range(ss):
        count = count + data[i]['dialog_length']

    #print(count)

    X = np.zeros((count, dim))
    text = []
    label = []
    m = 0
    
    # treat each turn as a data vector
    for i in range(ss):
        
        for row in data[i]['embedding_' + embedding_name]:
            X[m, :] = np.asarray(row)
            m=m+1
            
        for j in range(data[i]['dialog_length']):
            label.append(data[i]['labels'][j])
            text.append(data[i]['dialog_list'][j])
            
    return X, label, text

file = 'frames'
embedding = 'fasttext'
X, label, texts = read_data_sent_plus(file + '_3000', embedding, 'stop')

np.savez('data+label_'+file + '_'+ embedding +'.npz', data_matrix = X, labels = label, texts = texts)
