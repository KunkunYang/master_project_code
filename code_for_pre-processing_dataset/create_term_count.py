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

def read_data_word(dataset_name, embedding_name, a):
    
    # read and convert data from json file to matrix
    data = read_json_line('data/embed_'+ embedding_name + '_'+ a + '_data_'+ dataset_name +'.json')
    ss = np.size(data)

    dim = np.size(data[0]['embedding_' + embedding_name][0])

    # count total number of turns
    count = 0
    for i in range(ss):
        count = count + data[i]['dialog_length']

    #print(count)

    X = []
    label = []
    m = 0
    
    # treat each turn as a data vector
    for i in range(ss):
        
        for row in data[i]['dialog_list']:
            X.append(row)
            
        for j in range(data[i]['dialog_length']):
            label.append(data[i]['labels'][j])
            
    return X, label

file = 'frames'
embedding = 'glove'
texts_word, label = read_data_word(file + '_3000', embedding, 'stop')

# remove words that occur only once
frequency = defaultdict(int)

for text in texts_word:
    for token in text:
        frequency[token] += 1
        
#texts_word = [[token for token in text if frequency[token] >1] for text in texts_word]

dictionary =  corpora.Dictionary(texts_word)
ss = np.size(dictionary)
print('size: ', ss)

corpus = [dictionary.doc2bow(text) for text in texts_word]

X = np.zeros((3000,ss))

for i in range(3000):
    for j,k in corpus[i]:
        X[i, j] = k


a = dictionary.token2id
b = list(a.keys())


# create vocaubary
with open(file+'_emb_count.txt', 'w') as f:
    for i in range(np.size(b)):
        if i % 100 == 0:
            print(i)
            
        str_temp = b[i]
        
        for j in range(ss):
            if j == a[b[i]]:
                str_temp = str_temp + ' '+str(1)
            else:
                str_temp = str_temp + ' '+str(0)
        f.write(str_temp)
        f.write('\n')

f.close()


# create data with Count Model
np.savez('data+label_'+file + '_count.npz', data_matrix = X, labels = label, texts = texts_word)
