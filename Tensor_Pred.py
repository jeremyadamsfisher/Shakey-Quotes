"""
Modified from the code by Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
"""

import tensorflow as tf
from tensorflow.contrib import rnn
import os, pickle, sys
import numpy as np

def RNN(x, weights, biases):

    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    # Average Accuracy= 95.20% at 50k iter
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # 1-layer LSTM with n_hidden units but with lower accuracy.
    # Average Accuracy= 90.60% 50k iter
    # Uncomment line below to test but comment out the 2-layer rnn.MultiRNNCell above
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


if __name__ == '__main__':
    if (os.path.isfile('data/dict.pkl') and os.path.isfile('data/rev_dict.pkl')):
        print('Loading existing Pickle files')
        with open('data/dict.pkl', 'rb') as pkl_file:
            dictionary = pickle.load(pkl_file)
        with open('data/rev_dict.pkl', 'rb') as pkl_file:
            reverse_dictionary = pickle.load(pkl_file)
    else:
        print('Couldnt file the dictionary pickle files')

    with tf.Graph().as_default() as g:
        vocab_size = len(dictionary)

        # Parameters 
        n_input = 3
        n_hidden = 5 #500 should match the loaded model

        # tf Graph input
        x = tf.placeholder("float", [None, n_input, 1])
        y = tf.placeholder("float", [None, vocab_size])

        # RNN output node weights and biases
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([vocab_size]))
        }
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        pred = RNN(x, weights, biases)

    with tf.Session(graph=g) as session:
        saver.restore(session, 'my-model/mdl.ckpt')
        session.run(tf.global_variables_initializer())
        print("Model restored.")
        while True:
            prompt = "%s words: " % n_input
            sentence = input(prompt)
            sentence = sentence.strip()
            words = sentence.split(' ')
            if len(words) != n_input:
                continue
            try:
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                for i in range(8): # prepare a sentense of 8 words.
                    keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                    onehot_pred = session.run(pred, feed_dict={x: keys})
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                print(sentence)
            except:
                print("Word not in dictionary")
