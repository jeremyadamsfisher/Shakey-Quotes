import argparse
import nltk
from os import path, getcwd
import csv
import itertools
import pickle

if __name__ == "__main__":

    unknown_token = 'UNKNOWN_TOKEN'
    sentence_start_token = 'SENTENCE_START'
    sentence_end_token = 'SENTENCE_END'

    parser = argparse.ArgumentParser(description='Pickle word vector.')
    parser.add_argument('-f','--filepath', type=str, help='File path for csv.')
    parser.add_argument('-c','--csvcolumn', type=int, help='Csv column.')
    parser.add_argument('-n','--name', type=str, help='RNN name.')
    args = parser.parse_args()

    # Settings
    _VOCABULARY_SIZE = 5000
    _PICKLE_IDX_WRD_FILE = path.join(getcwd(), args.name + 'Idx2Word.pickle')
    _PICKLE_WRD_IDX_FILE = path.join(getcwd(), args.name + 'Word2Idx.pickle')

    # Download the NLTK libraries, which we only need to do once
    nltk.download('punkt')

    # Read the CSV file.
    sentences = []
    if args.filepath.endswith('csv'):
        print 'Reading CSV file...'
        with open(args.filepath, 'rU') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            # Split full comments into sentences
            sentences = itertools.chain(*[nltk.sent_tokenize(x[args.csvcolumn].decode('utf-8').lower()) for x in reader])
            # Append SENTENCE_START and SENTENCE_END
            sentences = ['%s %s %s' % (sentence_start_token, x, sentence_end_token) for x in sentences]

    print 'Parsed {} sentences.'.format(len(sentences))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print 'Found {} unique words tokens.'.format(len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(_VOCABULARY_SIZE-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    # Pickle the word vectors
    print 'Pickling most common {} tokens'.format(_VOCABULARY_SIZE)
    with open(_PICKLE_IDX_WRD_FILE, 'wb') as pkl_file1:
        pickle.dump(index_to_word, pkl_file1, protocol=pickle.HIGHEST_PROTOCOL)
    with open(_PICKLE_WRD_IDX_FILE, 'wb') as pkl_file2:
        pickle.dump(word_to_index, pkl_file2, protocol=pickle.HIGHEST_PROTOCOL)

    print 'Done.'
