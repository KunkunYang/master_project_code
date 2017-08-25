import json
import numpy as np
import nltk
import difflib
from nltk.corpus import stopwords

def read_json(filename):

    with open(filename, 'r') as f:
        data = json.load(f)

    return data

def read_json_line(filename):

    data = []
    
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    return data

def tokenize_nopunc_lower(file_content):
    words = nltk.word_tokenize(file_content)
    words =[word.lower() for word in words if word.isalpha()]
    # stop word
    stop = set(stopwords.words('english'))
    words = [word for word in words if word not in stop]
    
    return words

    
def create_vocab(filename):
    data = read_json_line(filename)
    ss = np.size(data)

    # extract only the text data
    with open(filename + '_sent_stop.txt', 'w') as f:
        for i in range(ss):
            for j in range(data[i]['dialog_length']):
                f.write(data[i]['dialog_list'][j] + ' ')
                data[i]['dialog_list'][j] = tokenize_nopunc_lower(data[i]['dialog_list'][j])

    with open('tokenized_stop_' + filename, 'w') as f:
        json.dump(data, f)

    # tokenize and create vocabulary
    file_content = open(filename + '_sent_stop.txt').read()

    a = tokenize_nopunc_lower(file_content)
    a = set(a)
    a = list(a)

    with open(filename + '_vocab_stop.txt', 'w') as f:
        for i in range(np.size(a)):
            f.write(a[i]+ ' ')



def embedding_and_OOV_words_fasttext(filename):

    # load vocabulary of data
    f1 = open(filename + '_vocab_stop.txt', 'r')
    raw1 = f1.read()

    words = raw1.split(' ')
    sizes = np.size(words)
    words = words[0:sizes-1] # remove spaces
    f1.close()

    # find embedding of each word in vocabulary
    # store OOV words
    
    f = open(filename + '_fasttext_OOV_stop.txt', 'w')
    g = open(filename + '_fasttext_emb_stop.txt', 'w')
            
    for index in range(101):
        print(index)
        
        array = []
        
        with open("wiki_en_" + str(index+1) + ".txt", "r",encoding="latin-1") as ins:
            for line in ins:
                array.append(line)
            
            for i in range(0,np.size(array)):
                if(array[i].split(' ')[0] in words)==True:
                    # add to vocabulary embedding
                    words.remove(array[i].split(' ')[0])
                    g.write(array[i])

    # add words not found to OOV file
    for i in range(0, np.size(words)):
       f.write(words[i] + ' ')

    print('total num of words: ', sizes-1)
    print('num of OOV words: ', np.size(words))
    print('OOV rate: ', np.size(words)/(sizes-1))
    
    f.close()
    g.close()
    ins.close()

def find_nearest_word_fasttext(filename):

    # generate list of all words in fasttext
    words = []
 
    for index in range(101):
        print(1, index)

        array = []
        
        with open("wiki_en_" + str(index+1) + ".txt", "r",encoding="latin-1") as ins:
            for line in ins:
                array.append(line)
            
            for i in range(0,np.size(array)):
                words.append(array[i].split(' ')[0])

    # read OOV words
    f = open(filename + '_fasttext_OOV_stop.txt').read()
    OOVs = f.split(' ')
    OOVs = OOVs[0:np.size(OOVs)-1]

    OOV_close = []
    
    for i in range(np.size(OOVs)):
        print('find ', i)
        # get closest match
        close = difflib.get_close_matches(OOVs[i], words, n = 1)
        if close == []:
            OOV_close.append('***notinfile***')
        else:
            OOV_close.append(close[0])

    g = open(filename + '_fasttext_emb_extra_stop.txt', 'a' )
    
    print(OOVs)
    print(OOV_close)
    
    for index in range(101):
        print(2, index)
        
        array = []
        
        with open("wiki_en_" + str(index+1) + ".txt", "r",encoding="latin-1") as ins:
            for line in ins:
                array.append(line)
            
            for i in range(0,np.size(array)):
                if(array[i].split(' ')[0] in OOV_close)==True:
                    sp = array[i].split(' ')
                    
                    # all index
                    OOV_in = [i for i, x in enumerate(OOV_close) if x == sp[0]]

                    for j in range(np.size(OOV_in)):
                        print(OOVs[OOV_in[j]])
                        sp[0] = OOVs[OOV_in[j]]
                        emb = ' '.join(sp)

                        g.write(emb)


    if ('***notinfile***' in OOV_close)==True:
        print('zero vec')
        OOV_in = [i for i, x in enumerate(OOV_close) if x == '***notinfile***']

        for j in range(np.size(OOV_in)):
            print(OOVs[OOV_in[j]])
            sp[0] = OOVs[OOV_in[j]]

            str_x = '0.'
            for gg in range(299):
                str_x = str_x + ' 0.'
            emb = sp[0] + ' ' + str_x

            g.write(emb)
            g.write('\n')
            
def combine_embeddings(filename):
    with open(filename + '_fasttext_emb_stop.txt', 'a' ) as f:
        with open(filename + '_fasttext_emb_extra_stop.txt', 'r' ) as g:
            for line in g:
                f.write(line)
    
def write_embedding(filename):

    # read embedding
    emb_word= []
    emb_array= []
    
    with open(filename + '_fasttext_emb_stop.txt', "r") as f:
        for line in f:
            sp = line.split(' ')
            emb_word.append(sp[0])
            emb_array.append([float(i) for i in sp[1:301]])
        
        
    data = read_json('tokenized_stop_' + filename)
    ss = np.size(data)
    
    for i in range(ss):
        print(i)
        data[i]['embedding_fasttext'] = []
        
        for j in range(data[i]['dialog_length']):
            emb = np.zeros(300)
            
            for word in data[i]['dialog_list'][j]:
                num = emb_word.index(word)
                emb += emb_array[num]

            
            emb = emb / np.size(data[i]['dialog_list'][j])
            
            if np.size(data[i]['dialog_list'][j]) == 0:
                emb = np.zeros(300)
                
            data[i]['embedding_fasttext'].append(emb.tolist())
    
        
    # Writing JSON data
    with open('embed_fasttext_stop_' + filename, 'w') as f:
        for datapoint in data:
            f.write(json.dumps(datapoint))
            f.write('\n')
        
 
###########

filename = 'data_amazon_3000.json'

create_vocab(filename)
embedding_and_OOV_words_fasttext(filename)
find_nearest_word_fasttext(filename)
combine_embeddings(filename)
write_embedding(filename)
