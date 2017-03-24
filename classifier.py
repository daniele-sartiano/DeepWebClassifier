#!/usr/bin/env python

import sys
import os
import argparse
import logging
import multiprocessing

import pandas
import numpy
import math
from keras.layers import Dense, Activation, LSTM, SimpleRNN, GRU, Dropout, Input, Flatten, GlobalMaxPooling1D, Merge
from keras.layers.embeddings import Embedding
#from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils import np_utils

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix

from  utils.normalizer import normalize_line


class Splitter(object):
    MIN_LEN = 2

    def __init__(self, resource):
        self.vocabulary = set()
        for token in resource:
            self.vocabulary.add(token.strip().lower())

 
    def ngrams(self, word):
        for i in range(Splitter.MIN_LEN, len(word)+1):
            if word[:i] in self.vocabulary:
                found = word[:i]
                elems = [el for el in self.ngrams(word[i:])]
                yield found , elems


    def merge(self, prefix, structure):
        a = structure[0]
        if not structure[1]:
            yield prefix + ' ' + a if prefix else a
        for el in structure[1]:
            for e in self.merge(prefix + ' ' + a if prefix else a, el):
                yield e
                

    def split(self, word):
        for combinations in self.ngrams(word):
            for tokens in self.merge('', combinations):
                splitted = tokens.split()
                yield splitted, sum([len(tok)/float(len(tokens)) for tok in splitted])


class WebClassifier(object):
    def __init__(self, nb_classes, input_dim=-1, batch_size=32, epochs=20, activation='softmax', loss='categorical_crossentropy', optimizer='adam', file_model='model.json', embeddings_weights=None, embeddings_dim=50, embeddings_weights_domains=None, embeddings_dim_domains=50, input_dim_domains=0, max_sequence_length=1000, max_sequence_length_domains=20):
        self.input_dim = input_dim
        self.input_dim_domains = input_dim_domains
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.embeddings_weights = embeddings_weights
        self.embeddings_dim = embeddings_dim
        self.embeddings_weights_domains = embeddings_weights_domains
        self.embeddings_dim_domains = embeddings_dim_domains
        self.max_sequence_length = max_sequence_length
        self.max_sequence_length_domains = max_sequence_length_domains
        self._create_model()

    def _create_model(self):
        content = Sequential()
        content.add(Embedding(
            self.input_dim, 
            self.embeddings_dim, 
            weights=[self.embeddings_weights],
            input_length=self.max_sequence_length,
            trainable=True
        ))

        content.add(Convolution1D(nb_filter=1024, filter_length=5, border_mode='same', activation='relu'))
        content.add(GlobalMaxPooling1D())
        content.add(Dense(512, activation='relu'))

        #content.add(Dense(self.nb_classes, activation='softmax'))

        domain = Sequential()

        domain.add(Embedding(
            self.input_dim_domains,
            self.embeddings_dim_domains, 
            weights=[self.embeddings_weights_domains],
            input_length=self.max_sequence_length_domains,
            trainable=True
        ))

        domain.add(Convolution1D(nb_filter=128, filter_length=3, border_mode='same', activation='relu'))
        domain.add(GlobalMaxPooling1D())
        domain.add(Dense(32))

        self.model = Sequential()
        self.model.add(Merge([content, domain], mode='concat'))
        
        self.model.add(Dense(self.nb_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    def train(self, X_train, y_train, dev=None, validation_split=0.0):
        self.model.fit(X_train, y_train,
                       shuffle=False,
                       batch_size=self.batch_size, 
                       nb_epoch=self.epochs, 
                       validation_split=validation_split,
                       validation_data=dev,
                       callbacks=[
                           EarlyStopping(verbose=True, patience=5, monitor='val_loss'),
                           ModelCheckpoint('TestModel-progress', monitor='val_loss', verbose=True, save_best_only=True)
                       ])


    def evaluate(self, X_test, y_test):
        score, acc = self.model.evaluate(X_test, y_test, batch_size=self.batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)


    def predict(self, X_test):
        y = self.model.predict_classes(X_test)
        p = self.model.predict_proba(X_test)
        return y, p


    def saveModel(self):
        # serialize
        model_json = self.model.to_json()
        with open(self.file_model, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")


    def describeModel(self):
        self.model.summary()
        return 'embeddings size %s, epochs %s\n%s' % (self.embeddings_dim,  self.epochs, self.model.to_yaml())

    def modelSummary(self):
        self.model.summary()



class WebClassifierMLP(WebClassifier):


    def _create_model(self):
        self.model = Sequential()
        self.model.add(Embedding(
            self.input_dim, 
            self.embeddings_dim, 
            weights=[self.embeddings_weights],
            input_length=self.max_sequence_length,
            trainable=False
        ))

        self.model.add(Flatten())        

        self.model.add(Dense(self.nb_classes, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# class WebClassifierLSTM(WebClassifier):


#     def _create_model(self):
#         self.model = Sequential()
#         self.model.add(Embedding(
#             self.input_dim, 
#             self.embeddings_dim, 
#             weights=[self.embeddings_weights],
#             input_length=self.max_sequence_length,
#             trainable=False
#         ))

#         self.model.add(Flatten())        

#         self.model.add(Dense(self.nb_classes, activation='softmax'))

#         self.model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

from copy_reg import pickle
from types import MethodType

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


class Reader(object):
    def __init__(self, input, processes, splitter):
        self.input = input
        self.pool = multiprocessing.Pool(processes)
        self.splitter = splitter

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


    def read(self):
        for r in self.pool.imap_unordered(self.extract, self.input, 5):
            if r:
                yield r

    def extract(self, line):
        d, t, l = line.strip().split('\t')

        d_words = sorted([el for el in self.splitter.split(d[:-3])], key=lambda x:x[1], reverse=True)
        selected = [tok for words, th in d_words[:5] for tok in words]
        domain_words = ' '.join(selected) if len(selected) > 0 else d[:-3]
        content = ' '.join(normalize_line(t))

        for label in l.split(','):
            l = 0 if int(label) == 13 else int(label)
            return d, l, content, domain_words


pickle(MethodType, _pickle_method, _unpickle_method)


def load_vectors(f):
    embeddings_index = {}
    embeddings_size = None
    for line in f:
        if embeddings_size is None:
            embeddings_size = int(line.strip().split()[-1])
            continue
        values = line.split()
        word = values[0]
        coefs = numpy.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index, embeddings_size


def main():
    parser = argparse.ArgumentParser(description='WebClassifier')
    parser.add_argument('-mw', '--max-words', help='Max words', type=int, required=True)
    parser.add_argument('-msl', '--max-sequence-length', help='Max sequence length', type=int, required=True)
    parser.add_argument('-mwd', '--max-words-domains', help='Max words domains', type=int)
    parser.add_argument('-msld', '--max-sequence-length-domains', help='Max sequence length', type=int, required=True)
    parser.add_argument('-e', '--embeddings', help='Embeddings', type=str, required=True)
    parser.add_argument('-ed', '--embeddings-domains', help='Embeddings for domain', type=str, required=False)
    parser.add_argument('-epochs', '--epochs', help='Epochs', type=str, default=200)
    parser.add_argument('-batch', '--batch', help='# batch', type=int, default=16)

    args = parser.parse_args()

    numpy.random.seed(7)

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', level=logging.INFO)

    logging.info('Reading vocabulary')
    
    vocabulary = None

    if args.embeddings_domains:
        for w in open(args.embeddings_domains):
            # skipping first line
            if vocabulary is None:
                vocabulary = set()
                continue
            vocabulary.add(w.split(' ')[0].strip().lower())
        for w in ['di', 'a', 'da', 'in', 'su', 'il', 'lo', 'la', 'un', 'e', 'i', 'o', 'al', 'd', 'l', 'c']:
            vocabulary.add(w)
        splitter = Splitter(vocabulary)

    logging.info('Reading corpus')

    # reader = Reader(sys.stdin, 18, splitter)

    # texts = []
    # labels = []
    # domains = []
    
    # for domain, label, content, domain_words in reader.read():
    #     labels.append(label)
    #     texts.append(content)
    #     domains.append(domain_words)
        

    texts = []
    labels = []
    domains = []
    for line in sys.stdin:
        d, t, l = line.strip().split('\t')
        for label in l.split(','):
            l = 0 if int(label) == 13 else int(label)
            labels.append(l)
            texts.append(' '.join(normalize_line(t)))
            if splitter:
                d_words = sorted([el for el in splitter.split(d[:-3])], key=lambda x:x[1], reverse=True)
                selected = [tok for words, th in d_words[:3] for tok in words]
                print d, selected
                domains.append(' '.join(selected) if len(selected) > 0 else d[:-3])


    logging.info('collecting domains sequences')

    tokenizer_domains = Tokenizer(nb_words=args.max_words_domains, lower=False)
    tokenizer_domains.fit_on_texts(domains)

    sequences_domains = tokenizer_domains.texts_to_sequences(domains)
    sequences_domains = sequence.pad_sequences(sequences_domains, maxlen=args.max_sequence_length_domains)

    logging.info('collecting content sequences')

    tokenizer = Tokenizer(nb_words=args.max_words, lower=False)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    sequences = sequence.pad_sequences(sequences, maxlen=args.max_sequence_length)

    logging.info('Splitting corpus')

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(sequences)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(labels)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(sequences_domains)

    toSplit = int(len(sequences) * 0.1)

    nb_classes = len(set(labels))

    X_dev = sequences[0:toSplit]
    X_dev_domains = sequences_domains[0:toSplit]

    y_dev_orig = labels[0:toSplit]
    y_dev = np_utils.to_categorical(y_dev_orig, nb_classes)

    X_train = sequences[toSplit:]
    X_train_domains = sequences_domains[toSplit:]

    y_train = np_utils.to_categorical(labels[toSplit:], nb_classes)

    logging.info('Reading Embedings: using the file %s, max words %s, max sequence length %s' % (args.embeddings, args.max_words, args.max_sequence_length))

    # prepare embedding matrix
    embeddings_index, embeddings_size = load_vectors(open(args.embeddings))

    nb_words = min(args.max_words, len(tokenizer.word_index))
    embedding_matrix = numpy.zeros((nb_words + 1, embeddings_size))
    for word, i in tokenizer.word_index.items():
        if i > args.max_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    embeddings_index = None

    # embeddings matrix for domains
    embeddings_index_domains, embeddings_size_domains = load_vectors(open(args.embeddings_domains))

    nb_words_domains = min(args.max_words, len(tokenizer_domains.word_index))
    embedding_matrix_domains = numpy.zeros((nb_words_domains + 1, embeddings_size_domains))
    for word, i in tokenizer_domains.word_index.items():
        if i > args.max_words:
            continue
        embedding_vector = embeddings_index_domains.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix_domains[i] = embedding_vector


    embeddings_index_domains = None


    webClassifier = WebClassifier(
        nb_classes, 
        input_dim=nb_words+1, 
        embeddings_dim=embeddings_size, 
        embeddings_weights=embedding_matrix, 
        embeddings_dim_domains=embeddings_size_domains, 
        embeddings_weights_domains=embedding_matrix_domains, 
        max_sequence_length=args.max_sequence_length,
        max_sequence_length_domains=args.max_sequence_length_domains,
        input_dim_domains=nb_words_domains+1, 
        epochs=args.epochs, 
        batch_size=args.batch)
    
    logging.info(webClassifier.describeModel())

    X_train = [X_train, X_train_domains]

    webClassifier.train(X_train, y_train, validation_split=0.3)

    logging.info('Evalutaing the model')

    X_dev = [X_dev, X_dev_domains]
    webClassifier.evaluate(X_dev, y_dev) 

    y_pred, p = webClassifier.predict(X_dev)

    # predicted = open('predicted', 'w')
    # for yy in y_pred:
    #     print >> predicted, yy
    # predicted.close()


    #print(classification_report(numpy.argmax(y_test,axis=1), y_pred, target_names=['a', 'b']))

    webClassifier.modelSummary()
    logging.info('\n%s' % classification_report(y_dev_orig, y_pred))
    logging.info('*'*80)
    #print(confusion_matrix(numpy.argmax(y_test,axis=1), y_pred))
    logging.info('\n%s' % confusion_matrix(y_dev_orig, y_pred))



if __name__ == '__main__':
    main()
