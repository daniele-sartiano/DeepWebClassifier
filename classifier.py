#!/usr/bin/env python

import sys
import os
import argparse
import logging

import pandas
import numpy
import math
from keras.layers import Dense, Activation, LSTM, SimpleRNN, GRU, Dropout, Input, Flatten, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils import np_utils

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix

from  utils.normalizer import normalize_line

class WebClassifier(object):
    def __init__(self, input_dim, nb_classes, batch_size=32, epochs=20, activation='softmax', loss='categorical_crossentropy', optimizer='adam', file_model='model.json', embeddings_weights=None, embeddings_dim=50, max_sequence_length=1000):
        self.input_dim = input_dim
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.embeddings_weights = embeddings_weights
        self.embeddings_dim = embeddings_dim
        self.max_sequence_length = max_sequence_length
        self._create_model()

    def _create_model(self):
        self.model = Sequential()
        self.model.add(Embedding(
            self.input_dim, 
            self.embeddings_dim, 
            weights=[self.embeddings_weights],
            input_length=self.max_sequence_length,
            trainable=False
        ))

        self.model.add(Convolution1D(nb_filter=1024, filter_length=5, border_mode='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_length=5))

        # self.model.add(Convolution1D(nb_filter=512, filter_length=5, border_mode='same', activation='relu'))
        # self.model.add(MaxPooling1D(pool_length=5))

        # self.model.add(Convolution1D(nb_filter=512, filter_length=5, border_mode='same', activation='relu'))
        # self.model.add(MaxPooling1D(pool_length=35))

        # self.model.add(Flatten())

        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(512, activation='relu'))
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
    parser.add_argument('-e', '--embeddings', help='Embeddings', type=str, required=True)
    parser.add_argument('-epochs', '--epochs', help='Epochs', type=str, default=200)
    parser.add_argument('-batch', '--batch', help='# batch', type=int, default=16)

    args = parser.parse_args()

    numpy.random.seed(7)

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', level=logging.INFO)

    logging.info('Reading corpus')

    texts = []
    labels = []
    for line in sys.stdin:
        d, t, l = line.strip().split('\t')
        for label in l.split(','):
            l = 0 if int(label) == 13 else int(label)
            labels.append(l)
            texts.append(' '.join(normalize_line(t)))

    tokenizer = Tokenizer(nb_words=args.max_words, lower=False)
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    sequences = sequence.pad_sequences(sequences, maxlen=args.max_sequence_length)

    logging.info('Splitting corpus')

    rng_state = numpy.random.get_state()
    numpy.random.shuffle(sequences)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(labels)

    toSplit = int(len(sequences) * 0.1)

    nb_classes = len(set(labels))

    X_dev = sequences[0:toSplit]
    y_dev_orig = labels[0:toSplit]
    y_dev = np_utils.to_categorical(y_dev_orig, nb_classes)

    X_train = sequences[toSplit:]
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

    
    webClassifier = WebClassifier(nb_words+1, nb_classes, embeddings_dim=embeddings_size, embeddings_weights=embedding_matrix, epochs=args.epochs, batch_size=args.batch)
    
    logging.info(webClassifier.describeModel())

    webClassifier.train(X_train, y_train, validation_split=0.3)

    logging.info('Evalutaing the model')

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
