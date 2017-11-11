#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
import pickle
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano

_MODEL_FILE = 'data/rnn-theano-80-5000-2017-11-10-08-01-06.npz'
_SRC_DATA = 'data/Shakespeare_Tragedies.csv'
_PICKLE_IDX_WRD_FILE = 'data/Idx_Wrd.pickle'
_PICKLE_WRD_IDX_FILE = 'data/Wrd_Idx.pickle'

_VOCABULARY_SIZE = 8000
_HIDDEN_DIM = 80

unknown_token = 'UNKNOWN_TOKEN'
sentence_start_token = 'SENTENCE_START'
sentence_end_token = 'SENTENCE_END'

def read_csv(csv_file):
    if csv_file.endswith('csv'):
        print 'Reading CSV file...'
        with open(csv_file, 'rU') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            # Split full comments into sentences
            sentences = itertools.chain(*[nltk.sent_tokenize(x[5].decode('utf-8').lower()) for x in reader])
            # Append SENTENCE_START and SENTENCE_END
            sentences = ['%s %s %s' % (sentence_start_token, x, sentence_end_token) for x in sentences]

    print 'Parsed %d sentences.' % (len(sentences))
    return sentences

if (os.path.isfile(_PICKLE_IDX_WRD_FILE) and os.path.isfile(_PICKLE_WRD_IDX_FILE)):
    print 'Loading exisitng Pickle files'
    with open(_PICKLE_IDX_WRD_FILE, 'rb') as pkl_file:
        index_to_word = pickle.load(pkl_file)
    with open(_PICKLE_WRD_IDX_FILE, 'rb') as pkl_file:
        word_to_index = pickle.load(pkl_file)
else
    # Read the CSV file.
    sentences = read_csv(_SRC_DATA)
    
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print 'Found %d unique words tokens.' % len(word_freq.items())

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(_VOCABULARY_SIZE-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    with open(_PICKLE_IDX_WRD_FILE, 'wb') as pkl_file:
        pickle.dump(index_to_word, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(_PICKLE_WRD_IDX_FILE, 'wb') as pkl_file:
        pickle.dump(word_to_index, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

model = RNNTheano(_VOCABULARY_SIZE, hidden_dim=_HIDDEN_DIM)
load_model_parameters_theano(_MODEL_FILE, model)

def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 250
senten_min_length = 7

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent).replace(' ,', ',').capitalize()
