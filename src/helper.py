from gensim.models.phrases import Phrases, Phraser
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import datetime
from rnn import rnn_data_generator
from sklearn.metrics import f1_score



def data_split(data_dict, labels, test_size=0.2):
    '''
        split data into training and test set.

        input

        data_dict -> data dictionary {'doc_id', 'text'}
        labels -> label dictionary {'doc_id', 'label'}

        return

        test_data_dict -> test data {'doc_id', 'text'}
        train_data_dict -> train data {'doc_id', 'text'}
        test_label_dict -> test label
        train_label_dict -> train label
    '''

    data_ids = list(data_dict.keys())
    train_ids, test_ids = train_test_split(data_ids, test_size=test_size)

    train_data_dict = data_dict
    test_data_dict = {}
    test_label_dict = {}
    train_label_dict = labels

    for ID in test_ids:
        test_data_dict[ID] = train_data_dict[ID]
        test_label_dict[ID] = train_label_dict[ID]

        del train_data_dict[ID]
        del labels[ID]

    return train_data_dict, test_data_dict, train_label_dict, test_label_dict




def phrase_trainer(n, sentences_list, num_doc, save_path):
    '''
        Train a n-gram model

        input:

        n -> n of the n-gram
        sentences_list -> list of all sentences in the whole training documents
        num_doc -> total number of the documents

        return:

        ngram_list -> list of ngram model [bigram, trigram ...]
    '''
    
    ngram_list = []

    # Training n-gram
    sentence_list_temp = sentences_list
    print()
    for i in range(n-1):
        print('Training n-gram:\t', i+2)
        start_time = datetime.datetime.now()
        phrases = Phrases(sentence_list_temp, min_count=num_doc*0.01)
        ngram = Phraser(phrases)
        sentence_list_temp = ngram[sentence_list_temp]
        ngram_list.append(ngram)
        ngram.save(save_path+str(i))
        print('Total time used:\t', datetime.datetime.now() - start_time)

    return ngram_list




def get_word_index(word_list):

    '''
        Word indexing

        input:

        word_list -> ['string1', 'string2', 'string3', ...]

        return:

        word_index -> {'string1':index_num, 'string2':index_num, ...}

    '''

    word_index = {}
    cur_index_num = 0
    for word in word_list:
        if word not in word_index:
            word_index[word] = cur_index_num
            cur_index_num += 1
    
    return word_index





def get_weight_matrix(word_index, w2v_model):

    '''
        Creating embedding matrix using word indexing

        input

        word_index -> {'word': 1, 'word2':2, ...}

        return

        weight_matrix -> numpy array with shape (num_word, vector_len)
        
    '''

    vocab_size = len(word_index) + 1
    weight_matrix = np.zeros((vocab_size, w2v_model.vector_size))

    for word, i in word_index.items():
        try:
            weight_matrix[i] = w2v_model.wv[word]
        except KeyError:
            continue

    return weight_matrix





def read_data(doc_dir, num_data=1000, is_test=False):

    '''
        Read the documents, re-assign new ids and replace old ids

        Input:

        doc_dir ->  path of document
        num_data -> number of data to read

        Return:

        data_dict -> dictionary stored training data
        label_dict -> dictionary stored training label

    '''

    df = pd.read_csv(doc_dir)
    max_data = df.shape[0]

    if num_data >= max_data:
        num_data = max_data

    data = df.loc[:num_data-1].to_dict()
    data_dict = data['question_text']

    if not is_test:
        label_dict = data['target']
    else:
        label_dict = {}

    return data_dict, label_dict




def get_f1(test_data, test_label, model,word_index, ngram_models, max_seq_len, 
            batch_size=1000, shuffle=False, threadhold=(0.3, 0.6), threadhold_step=0.01):
    
    test_generator = rnn_data_generator(test_data, test_label, word_index, 
                                    ngram_models, batch_size=batch_size, shuffle=False, 
                                    max_len=max_seq_len, is_test=True)

    pre_array = np.empty((0,2))
    true_array = np.empty((0,2))

    for i in range(len(test_generator)):
        data, labels, test_id = test_generator[i]
        test_pre = model.predict(data)
        pre_array = np.vstack((pre_array, test_pre))
        true_array = np.vstack((true_array, labels))

    f1_dict = {}
    for t in np.arange(threadhold[0], threadhold[1], threadhold_step):
        f1_dict[t] = f1_score(true_array.argmax(axis=1), (pre_array[:,1] > t).astype(int))
        print(float(t), f1_dict[t])

    return f1_dict