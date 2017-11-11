#! /usr/bin/env python

import csv
import itertools
import operator
import pickle
from utils import *
from rnn_theano import RNNTheano
import argparse
import os

# Settings
vocabulary_size = 5000
hidden_dim = 80

Tragedy_RNN = 'data/TragedyMind.npz'
Tragedy_PICKLE_IDX_WRD_FILE = 'data/TragedyMindIdx2Word.pickle'
Tragedy_PICKLE_WRD_IDX_FILE = 'data/TragedyMindWord2Idx.pickle'

Comedy_RNN = 'data/ComedyMind.npz'
Comedy_PICKLE_IDX_WRD_FILE = 'data/ComedyMindIdx2Word.pickle'
Comedy_PICKLE_WRD_IDX_FILE = 'data/ComedyMindWord2Idx.pickle'

unknown_token = 'UNKNOWN_TOKEN'
sentence_start_token = 'SENTENCE_START'
sentence_end_token = 'SENTENCE_END'


def generate_quote_wrapper(RNN, Idx2Word_pickle, Word2Idx_pickle):
    if (os.path.isfile(Idx2Word_pickle) and os.path.isfile(Word2Idx_pickle)):
        print 'Loading Pickled Word vector files...'
        with open(Idx2Word_pickle, 'rb') as pkl_file:
            index_to_word = pickle.load(pkl_file)
        with open(Word2Idx_pickle, 'rb') as pkl_file:
            word_to_index = pickle.load(pkl_file)
    else:
        raise NameError('Could not find pickled word vector!')

    print 'Loading model...'
    model = RNNTheano(vocabulary_size, hidden_dim=hidden_dim)
    load_model_parameters_theano(RNN, model)

    def clean_up_sentence(sent):
        clean_dict = {' ,': ',',
                      ' !': '!',
                      ' :': ':',
                      ' ?': '?',
                      ' .': '.',
                      ' \'': '\'',
                      ' --': '--'}

        for bad, good in clean_dict.iteritems():
            sent = sent.replace(bad, good)

        pms = clean_dict.values()

        for d_pm in itertools.product(pms, pms):
            d_pm = str(d_pm[0]) + str(d_pm[1])
            sent = sent.replace(d_pm, '')

        for pm in pms:
            if sent.startswith(pm):
                sent = sent[1:]

        sent = sent.capitalize()

        return(sent)

    def generate_sentence(senten_min_length):
        sentence_array = []

        while len(sentence_array) < senten_min_length:
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
            sentence_array = [index_to_word[x] for x in new_sentence[1:-1]]

        sentence_str = ' '.join(sentence_array)
        return clean_up_sentence(sentence_str)

    def generate_subquote():
        new_sentence = generate_sentence(7)
        if new_sentence.endswith('.') or new_sentence.endswith('?') or new_sentence.endswith('!'):
            return new_sentence
        else:
            new_sentence = new_sentence + '\n' + generate_subquote()
            return new_sentence

    def generate_quote():
        while True:
            new_quote = generate_subquote()
            if 4 >= len(new_quote.split('\n')) > 2:
                return new_quote

    return generate_quote()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pickle word vector.')
    parser.add_argument('-t', '--type', type=str, help='Play type for quote.')
    args = parser.parse_args()

    if args.type.lower() == 'comedy':
        print generate_quote_wrapper(Comedy_RNN, Comedy_PICKLE_IDX_WRD_FILE, Comedy_PICKLE_WRD_IDX_FILE)

    elif args.type.lower() == 'tragedy':
        print generate_quote_wrapper(Tragedy_RNN, Tragedy_PICKLE_IDX_WRD_FILE, Tragedy_PICKLE_WRD_IDX_FILE)
