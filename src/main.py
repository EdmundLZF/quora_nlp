from wordProcess import wordProcess
from w2v import w2v
from helper import read_data, phrase_trainer, get_word_index, get_weight_matrix, data_split, get_f1
import datetime
import multiprocessing
from multiprocessing.pool import ThreadPool
from gensim.utils import grouper
import gensim
from cnn import cnn_data_generator, cnn
import os
from functools import partial
import numpy as np
import pandas as pd
from rnn import rnn_data_generator, rnn


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
num_pool = multiprocessing.cpu_count()
print('\nTotal number of processor', num_pool)
print(os.getcwd())



########################################
##  Loading data
########################################



print('\n########## Loading data ##########')


# Text processing
train_data_dir = './data/train.csv'
num_data_read = 10000000

## Load txt
start_time = datetime.datetime.now()

train_data_dict, train_label_dict = read_data(train_data_dir, num_data=num_data_read)


print('\nTotal time for loading data', datetime.datetime.now() - start_time)
print('Number of train label\t', len(train_label_dict))
print('Number of train data\t', len(train_data_dict))



########################################
##  Training n-gram
########################################



print('\n########## Training n-gram ##########')


## Train n-gram model
start_time = datetime.datetime.now()


def ngram_train_worker(train_list):
    # multiprocessing worker for training ngram
    l = []
    for _doc_id, text in train_list:
        wp = wordProcess(text)
        l += wp.text

    return l


def training_ngram(train_data_dict, num_pool, num_ngram, ngram_num_doc, ngram_model_path):

    with multiprocessing.Pool(processes=num_pool) as pool:
        group_itr = grouper(train_data_dict.items(), int(len(train_data_dict.items())/num_pool))
        list_tmp = pool.map(ngram_train_worker, group_itr)

    ngram_train_list = []
    for sen_list in list_tmp:
        ngram_train_list += sen_list

    ngram_models = phrase_trainer(num_ngram, ngram_train_list, ngram_num_doc, ngram_model_path)

    return ngram_models, ngram_train_list


# Training n-gram
ngram_parameter = {'num_pool': num_pool,
                    'num_ngram': 1,
                    'ngram_num_doc': len(train_data_dict),
                    'ngram_model_path': './models/ngram/'
                }


ngram_models, train_list_sentences = training_ngram(train_data_dict, **ngram_parameter)

print('\nTotal time for training ngram models', datetime.datetime.now() - start_time)
print('Number of sentence to train\t', len(train_list_sentences))
print('Number of ngram models\t', len(ngram_models))



########################################
##  Train or load Word2vec model
########################################



print('\n########## Train Word2vec model ##########')


start_time = datetime.datetime.now()

def convet_ngram_worker(train_list_sentences, ngram_models=None):
    # multiprocessing worker extracting and converting n-gram from text data
    train_data = train_list_sentences
    for n in ngram_models:
        train_data = list(n[train_data])

    return train_data



def convet_to_ngram(train_list_sentences, ngram_models, num_pool):

    with multiprocessing.Pool(processes=num_pool) as pool:
        group_itr = grouper(train_list_sentences, int(len(train_list_sentences)/num_pool))
        list_tmp = pool.map(partial(convet_ngram_worker, ngram_models=ngram_models), group_itr)

    train_data = []
    for l in list_tmp:
        train_data += l

    return train_data



# Training word2vec model
w2v_train_data = convet_to_ngram(train_list_sentences, ngram_models, num_pool)



# w2v_parameter = {'save_path': './models/word2vec/w2v_model.model', 
#                 'load_path':None, 
#                 'n_processor':num_pool, 
#                 'load_model':False, 
#                 'size':300, 
#                 'window':5, 
#                 'min_count':1, 
#                 'iter':100}
# w2v_model = w2v(w2v_train_data, **w2v_parameter)
# w2v_trained_model = w2v_model.train_model()
# w2v_model.save_model()



# Loading word2vec model
w2v_parameter = {'save_path': './models/word2vec/w2v_model.model', 
                'load_path':'./models/word2vec/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin', 
                'n_processor':num_pool, 
                'load_model':True, 
                'size':300, 
                'window':5, 
                'min_count':1, 
                'iter':100}

w2v_model = w2v(w2v_train_data, **w2v_parameter)
w2v_trained_model = w2v_model.model

print('\nTotal time for training wv2 models\t', datetime.datetime.now() - start_time)
print('The size of the models\t', len(w2v_trained_model.wv.vocab))



########################################
##  Creating embedding matrix
########################################



print('\n########## Creating embedding matrix ##########')


start_time = datetime.datetime.now()
word_list = [item for sublist in w2v_train_data for item in sublist]
word_index = get_word_index(word_list)
embedding_matrix = get_weight_matrix(word_index, w2v_trained_model)
del w2v_model, w2v_trained_model, w2v_train_data

print('\nTotal time for Creating embedding matrix\t', datetime.datetime.now() - start_time)
print('The shape of embedding matrix\t',embedding_matrix.shape)



########################################
##  Simply CNN model
########################################



print('\n########## CNN model Training ##########')


max_seq_len = 100
lstm_len = 60
cnn_model_path = './models/cnn_model/'
batch_size = 10000
start_time = datetime.datetime.now()

# split data into test and training set
cnn_train_data, cnn_test_data, cnn_train_label, cnn_test_label = data_split(train_data_dict, 
                                                                            train_label_dict, 
                                                                            test_size=0.2)

# create data generator for cnn model

training_generator = cnn_data_generator(cnn_train_data, cnn_train_label, word_index, 
                                        ngram_models, batch_size=batch_size, shuffle=True, max_len=max_seq_len)
validation_generator = cnn_data_generator(cnn_test_data, cnn_test_label, word_index, 
                                        ngram_models, batch_size=batch_size, shuffle=True, max_len=max_seq_len)

model_parameter = {'n_classes':2, 
                    'n_data':len(word_index) + 1, 
                    'n_feature':300, 
                    'embedding_matrix':embedding_matrix, 
                    'max_seq_len':max_seq_len}

cnn_model = cnn(**model_parameter)
cnn_simply_model = cnn_model.simply_cnn()


print('\n\n########## Training simply CNN model ##########\n\n')
cnn_simply_model.summary()
cnn_simply_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=10,
                    steps_per_epoch=(len(cnn_train_data) // batch_size),
                    validation_steps=(len(cnn_test_data) // batch_size),
                    use_multiprocessing=False,
                    workers=1)


print('\nTotal time for training model\t', datetime.datetime.now() - start_time)
cnn_simply_model.save(cnn_model_path+'cnn_simply_model.h5')
print('Model save to: \n', cnn_model_path+'cnn_simply_model.h5')

del training_generator, validation_generator




print('\nGetting f1:\n')
f1_dict = get_f1(cnn_test_data, cnn_test_label, cnn_simply_model, word_index,
                ngram_models, max_seq_len, batch_size=1000, shuffle=True)


print('\nTotal time for calculating f1\t', datetime.datetime.now() - start_time)
print('F1 score\t', f1_dict)
del cnn_simply_model



########################################
##  Simply LSTM model
########################################



print('\n\n########## Training Simply LSTM model ##########\n\n')
start_time = datetime.datetime.now()
rnn_training_generator = rnn_data_generator(cnn_train_data, cnn_train_label, word_index, 
                                        ngram_models, batch_size=batch_size, shuffle=True, max_len=max_seq_len)
rnn_validation_generator = rnn_data_generator(cnn_test_data, cnn_test_label, word_index, 
                                        ngram_models, batch_size=batch_size, shuffle=True, max_len=max_seq_len)

rnn_model_path = './models/rnn_model/'
rnn_model_parameter = {'n_classes':2, 
                    'n_data':len(word_index) + 1, 
                    'n_feature':300, 
                    'embedding_matrix':embedding_matrix, 
                    'max_seq_len':max_seq_len,
                    'lstm_len':lstm_len}


rnn_model = rnn(**rnn_model_parameter)
rnn_simply_lstm_model = rnn_model.simply_rnn_lstm()


rnn_simply_lstm_model.summary()
rnn_simply_lstm_model.fit_generator(generator=rnn_training_generator,
                    validation_data=rnn_validation_generator,
                    epochs=1,
                    steps_per_epoch=(len(cnn_train_data) // batch_size),
                    validation_steps=(len(cnn_test_data) // batch_size),
                    use_multiprocessing=False,
                    workers=1)



print('\nTotal time for training model\t', datetime.datetime.now() - start_time)
rnn_simply_lstm_model.save(rnn_model_path+'rnn_simply_lstm_model.h5')
print('Model save to: \n', rnn_model_path+'rnn_simply_lstm_model.h5')



print('\nGetting f1:\n')
rnn_simply_lstm_f1 = get_f1(cnn_test_data, cnn_test_label, rnn_simply_lstm_model, word_index,
                            ngram_models, max_seq_len, batch_size=1000, shuffle=True)
print('\nTotal time for calculating f1\t', datetime.datetime.now() - start_time)
print('F1 score\t', rnn_simply_lstm_f1)
del rnn_simply_lstm_model



########################################
##  Bidrectinal LSTM model
########################################



print('\n\n########## Training bidrectinal LSTM model ##########\n\n')
start_time = datetime.datetime.now()

rnn_bid_lstm_model = rnn_model.bidrectinal_lstm()
rnn_bid_lstm_model.summary()
rnn_bid_lstm_model.fit_generator(generator=rnn_training_generator,
                                validation_data=rnn_validation_generator,
                                epochs=1,
                                steps_per_epoch=(len(cnn_train_data) // batch_size),
                                validation_steps=(len(cnn_test_data) // batch_size),
                                use_multiprocessing=False,
                                workers=1)


print('\nTotal time for training model\t', datetime.datetime.now() - start_time)
rnn_bid_lstm_model.save(rnn_model_path+'rnn_bid_lstm_model.h5')
print('Model save to: \n', rnn_model_path+'rnn_bid_lstm_model.h5')


print('\nGetting f1:\n')
rnn_bid_lstm_f1 = get_f1(cnn_test_data, cnn_test_label, rnn_bid_lstm_model, word_index,
                        ngram_models, max_seq_len, batch_size=1000, shuffle=True)
print('\nTotal time for calculating f1\t', datetime.datetime.now() - start_time)
print('F1 score\t', rnn_bid_lstm_f1)