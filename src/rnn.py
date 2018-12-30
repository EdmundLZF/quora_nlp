import numpy as np
import keras
from keras.layers import Dense, Input, Flatten
from keras.layers import LSTM, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, Concatenate, Bidirectional
from keras.models import Model
from keras.optimizers import RMSprop
import gensim
from wordProcess import wordProcess
from keras.callbacks import Callback
import tensorflow as tf
from keras import backend as be
import multiprocessing
from multiprocessing.pool import ThreadPool
from gensim.utils import grouper



class rnn:
    
    def __init__(self, n_classes, n_data, n_feature, embedding_matrix, max_seq_len, lstm_len):

        '''
            Initializing cnn embeding model
        '''
        
        config = tf.ConfigProto(device_count={"CPU": 8})
        be.tensorflow_backend.set_session(tf.Session(config=config))
        
        self.n_classes = n_classes
        self.embedding_layer = Embedding(n_data,
                            n_feature,
                            weights=[embedding_matrix],
                            input_length=max_seq_len,
                            trainable=True)

        self.sequence_input = Input(shape=(max_seq_len,))
        self.embedded_sequences = self.embedding_layer(self.sequence_input)
        self.lstm_len = lstm_len

    

    def simply_rnn_lstm(self):

        layer = LSTM(self.lstm_len)(self.embedded_sequences)
        layer = Dense(256)(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(self.n_classes)(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=self.sequence_input, outputs=layer)

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['mae', 'acc'])

        return model
        

    def bidrectinal_lstm(self):
        
        layer = Bidirectional(LSTM(self.lstm_len))(self.embedded_sequences)
        layer = Dropout(0.5)(layer)
        layer = Dense(self.n_classes, activation='sigmoid')(layer)
        model = Model(input=self.sequence_input, outputs=layer)

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['mae', 'acc'])

        return model




class rnn_data_generator(keras.utils.Sequence):

    def __init__(self, data, labels, word_index, phraser_list, batch_size=32, 
                shuffle=True, max_len=100, is_test=False):

        '''
            Class initialization
            
            Input:

            data -> dictionary store doc_IDs and text. {'doc_IDs':test_string, ...}
            labels -> dictionary store doc_IDs and labels. {'doc_IDs':'labels', ...}
            word_index -> dicionary store word and index number. {'word':index_num, ...}
            phraser_list -> list of n-gram model [bigram, trigram, ...]
            batch_size -> number of data for each batch
            shuffle -> shuffle data before each generation
            max_len -> max length of the sequence
        '''

        self.data = {}
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.phraser_list = phraser_list
        self.max_len = max_len
        self.word_index = word_index
        self.is_test = is_test
        self.num_classes = self.__get_num_classes()
        self.__processing_data(data)
        self.list_IDs = list(self.data.keys())
        self.on_epoch_end()


    
    def __processing_data(self, data):

        '''
            Convert string into wordProcess object and convert to n-grams
        '''


        num_pool = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_pool) as pool:
            group_itr = grouper(data.items(), int(len(data.items())/num_pool))
            list_tmp = pool.map(self.generator_worker, group_itr)

        data_list = []
        for l in list_tmp:
            data_list += l

        self.data = dict(data_list)



    def generator_worker(self, data_items):

        '''
            worker for __processing_data method
        '''

        dict_tmp = []
        for doc_ids, text in data_items:
            wp = wordProcess(text, label=self.labels[doc_ids])
            wp.phrase(self.phraser_list)
            dict_tmp.append((doc_ids, wp))

        return dict_tmp



    def __get_num_classes(self):
        
        '''
            Determine number of unique classes in the data

            return:
            
            number of unique classes
        '''

        return len(set(self.labels.values()))



    def on_epoch_end(self):

        '''
            Create indexes of data and shuffle data in each epoch if specified
        '''

        self.indexes = np.arange(len(self.list_IDs))
        # shuffle data in each epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):

        '''
            Generate data for each epoch.

            input

            list_IDs_temp -> list of ID to generate in each epoch

            return

            X -> numpy array with shape (batch_size, max_len)
            y -> one-hot representation of the label of X with shape (batch_size, num_classes)
        '''

        X = np.zeros((self.batch_size, self.max_len))
        y = np.empty(self.batch_size)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            wp = self.data[ID]
            data_array, data_label = wp.to_sequences(self.word_index, self.max_len)
            X[i] += data_array

            # convert class from string to int and convert class number range from [1 to 23] to [0 to 22]
            # if the number in [1 to 23], the to_categorical method with create 24 class. because the method
            # will consider number of classes be the max number in the arrary if the max number greater than
            # num_classes
            y[i] = data_label

        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)



    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        if self.is_test:
            return X, y, list_IDs_temp

        else:
            return X, y



    def __len__(self):
        return int(np.floor(len(self.list_IDs)) / self.batch_size)
