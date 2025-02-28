"""
Modified from the code by Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
"""
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import collections, random, time, pickle

def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

def read_file(infile):
    print('Reading Input file...')
    file_cnt = 0
    with open(infile, 'rU') as f:
        recs = []
        for lines in f:
            file_cnt += 1
            #line = lines.strip().split(',')[-1].split()
            line = lines.strip().split()
            if len(line) > 6:
                recs.append(line)
        words = np.array(recs)
        print('training rows:',len(recs))
        print('Total File rows:',file_cnt)
    return words

def build_dicts(words):
    words = [item for sublist in words for item in sublist]
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

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
    start_time = time.time()

    graphs = './graphs'
    writer = tf.summary.FileWriter(graphs)
    
    training_data = read_file('data/STrag_Small.csv')
    
    dictionary, reverse_dictionary = build_dicts(training_data)
    with open('data/dict.pkl', 'wb') as pkl_file:
        pickle.dump(dictionary, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data/rev_dict.pkl', 'wb') as pkl_file:
        pickle.dump(reverse_dictionary, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    print('Number of words :',len(dictionary))
    vocab_size = len(dictionary)

    # Parameters
    learning_rate = 0.01
    training_iters = 500 # set to atleast a few thousands to get better accuracy
    display_step = 200 # set based on number records in the file
    n_input = 3

    # number of units in RNN cell
    n_hidden = 500 # set to atleast a few hundreds to get good predictions. Too high will lead to over fitting

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

    pred = RNN(x, weights, biases)

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Launch the graph
    with tf.Session() as session:
        session.run(init)
        step = 0
        offset = random.randint(0,n_input+1)
        end_offset = n_input + 1
        acc_total = 0
        loss_total = 0

        writer.add_graph(session.graph)
        while step <= training_iters:
            step += 1
            print('Iteration :',step)
            rec_cnt = 0
            for line in training_data:
                rec_cnt += 1
                # Generate a minibatch. Add some randomness on selection process.
                if offset > (len(line)-end_offset):
                    offset = random.randint(0, n_input+1)

                symbols_in_keys = [ [dictionary[ str(line[i])]] for i in range(offset, offset+n_input) ]
                symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

                symbols_out_onehot = np.zeros([vocab_size], dtype=float)
                symbols_out_onehot[dictionary[str(line[offset+n_input])]] = 1.0
                symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

                _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                        feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
                loss_total += loss
                acc_total += acc
                if (rec_cnt+1) % display_step == 0:
                    print("Iter= " + str(rec_cnt+1) + ", Average Loss= " + \
                          "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                          "{:.2f}%".format(100*acc_total/display_step))
                    acc_total = 0
                    loss_total = 0
                    symbols_in = [line[i] for i in range(offset, offset + n_input)]
                    symbols_out = line[offset + n_input]
                    symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred, 1).eval())]
                    print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))

        print("Optimization completed!")
        print("Elapsed time: ", elapsed(time.time() - start_time))
        saver.save(session, 'my-model/mdl.ckpt')
        # Do a few rounds of prediction
        while True:
            prompt = "%s words: " % n_input
            sentence = input(prompt)
            sentence = sentence.strip()
            words = sentence.split(' ')
            if len(words) != n_input:
                continue
            try:
                symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
                for i in range(8):
                    keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                    onehot_pred = session.run(pred, feed_dict={x: keys})
                    onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                    sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
                    symbols_in_keys = symbols_in_keys[1:]
                    symbols_in_keys.append(onehot_pred_index)
                print(sentence)
            except:
                print("Word not in dictionary")
