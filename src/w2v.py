from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from gensim.utils import grouper
import sys
import multiprocessing


class w2v:

    def __init__(self, data_list, save_path=None, load_path=None, n_processor=1, load_model=False, 
                size=300, window=5, min_count=1, iter=100):

        '''
            Word2Vec model class for handling Word2Vec model, including load model,
            train model and save model.

            input:

            data_list -> list of tokens. [['first','sentence'], ['second', 'sentence'], ...]
            save_path -> path for saving the model
            load_path -> path for loading the model
            n_processor -> defind number processor use for training model
            load_mode -> bool, load a pre-train model or a new model
            size -> number of feature for word2vec model
            window -> the window of the model
            min_count -> min number of time the word appear in the collection, in order
                        in order to take in count for training model
            iter -> number of iteration for training model
        '''

        self.data = data_list
        self.load_path = load_path
        self.save_path = save_path
        self.n_processor = n_processor
        self.is_load_model = load_model

        if self.is_load_model:
            self.model = self.load_model()
        else:
            self.model = Word2Vec(size=size, window=window, min_count=min_count, 
                                workers=self.n_processor, iter=iter)



    def load_model(self):

        '''
            Loading exist model. Only .bin or .model file can be load. 
            Otherwise, exit the program with 1.
        '''
        
        model_type = self.load_path.split('.')[-1]
        # Validate the model file type
        if model_type == 'bin':
            model = KeyedVectors.load_word2vec_format(self.load_path, binary=True)
        elif model_type == 'model':
            model = Word2Vec.load("word2vec.model")
        else:
            print('\nCannot load model file type.\n')
            sys.exit(1)

        return model



    def train_model(self):

        '''
            Training model by using multi-processing.

            return trained model
        '''

        if not self.is_load_model: # For increase the training new creating a new model
            self.model.build_vocab(self.data, progress_per=10000)

        # Defind the training worker function
        worker = lambda sents: self.model.train(sents, 
                                                epochs=self.model.iter, 
                                                total_examples=self.model.corpus_count)

        # Multiprocessing
        with multiprocessing.Pool(processes=self.n_processor) as pool:
            jobs = grouper(self.data, int(len(self.data)/self.n_processor))
            list_tmp = pool.imap(worker, jobs)

        return self.model



    def save_model(self):
        '''
            Save model.
        '''
        self.model.save(self.save_path)

