#!/usr/bin/env python

import sys
import os
import pandas

import numpy
import math
from keras.layers import Dense, Activation, LSTM, SimpleRNN, GRU, Dropout, Input
from keras.layers.embeddings import Embedding
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix

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
        self.model.add(LSTM(100))
        
        self.model.add(Dense(self.nb_classes))
        self.model.add(Activation(self.activation))

        self.model.compile(loss=self.loss,
              optimizer=self.optimizer,
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
        print 'class_weight', self.class_weight 
        print 'batch_size', self.batch_size
        print 'epochs', self.epochs
        print 'activation', self.activation
        print classifier.model.summary()



def load(f):
    embeddings_index = {}
    for line in f:
        values = line.split()
        word = values[0]
        coefs = numpy.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index


def main():
    MAX_WORDS = 20000
    MAX_SEQUENCE_LENGTH = 1000
    CORPUS = 'data/it-train'
    IT_VECTORS = 'data/it-vectors.txt'
    EMBEDDING_DIM = 50
    
    #from keras.datasets import reuters
    #(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=1000, test_split=0.2)
    
    tokenizer = Tokenizer(nb_words=MAX_WORDS)

    texts = []
    labels = []
    for i, line in enumerate(open(CORPUS)):
        d, l, t = line.strip().split('\t')
        labels.append(int(l))
        texts.append(t)

    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    toSplit = int(len(sequences) * 0.1)

    nb_classes = 14

    X_test = sequences[0:toSplit]
    y_test_orig = labels[0:toSplit]
    y_test = np_utils.to_categorical(y_test_orig, nb_classes)

    X_train = sequences[toSplit:]
    y_train = np_utils.to_categorical(labels[toSplit:], nb_classes)

    # prepare embedding matrix
    embeddings_index = load(open(IT_VECTORS))
    nb_words = min(MAX_WORDS, len(tokenizer.word_index))
    embedding_matrix = numpy.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i > MAX_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


    webClassifier = WebClassifier(nb_words+1, nb_classes, embeddings_weights=embedding_matrix, epochs=200)
    webClassifier.train(X_train, y_train, validation_split=0.3)
    webClassifier.evaluate(X_test, y_test) 

    y_pred, p = webClassifier.predict(X_test)

    predicted = open('predicted', 'w')
    for yy in y_pred:
        print >> predicted, yy
    predicted.close()

    #print(classification_report(numpy.argmax(y_test,axis=1), y_pred, target_names=['a', 'b']))
    print(classification_report(y_test_orig, y_pred))
    print '*'*80
    #print(confusion_matrix(numpy.argmax(y_test,axis=1), y_pred))
    print(confusion_matrix(y_test_orig, y_pred))


if __name__ == '__main__':
    main()
