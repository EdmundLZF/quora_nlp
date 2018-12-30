from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer, RegexpTokenizer
from gensim.models.phrases import Phrases, Phraser
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np




class wordProcess:

    def __init__(self, text, label=None, keep_punc=False):
        '''
            Input:

            text -> text of the document, string
            label -> label of the documents, string. Set None if no label.
        '''

        self.label = label

        # TODO: takenizing text in [[the, first, sentence], [the, second, sentence]] format
        sent = sent_tokenize(text) # Extract all sentence into a list
        self.text = []

        if not keep_punc:
            tw = RegexpTokenizer(r'\w+')
            tokenizer = lambda s : tw.tokenize(s)
        else:
            tokenizer = lambda s : word_tokenize(s)

        for s in sent:
            tokens = tokenizer(s)
            self.text += [tokens]



    def phrase(self, phraser_list):
        '''
            Extract and convert all n-gram in the text

            input:

            phraser_list -> list of phraser [bigram_phraser, trigram_phraser, ...]
        '''

        # TODO Extract n-gram
        sentence_list_temp = self.text

        for p in phraser_list:
            sentence_list_temp = p[sentence_list_temp]

        self.text = sentence_list_temp



    def to_sequences(self, word_index, max_len):

        '''
            Convert text to sequences representation.

            input

            word_index -> word indexing dictionary {'word1':index_number, ...}
            max_len -> max len of each sequences.

            return

            word_seq -> the sequences representation of the text. [0, 2, 323, 34, ...]
            label -> the label of the document

        '''

        # Conbine all sentence into a single list
        flattened_list = [y for x in self.text for y in x]

        # trim the list if the len of list over the max len
        if len(flattened_list) > max_len:
            flattened_list = flattened_list[:max_len]

        # convert to sequences and padding with zero
        word_seq = [word_index[word] for word in flattened_list] + [0] * (max_len - len(flattened_list))

        self.word_seq = word_seq

        return self.word_seq, self.label