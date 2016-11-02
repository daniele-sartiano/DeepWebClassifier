#!/usr/bin/env python

import sys
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
    def __init__(self, input_dim, nb_classes, batch_size=32, epochs=20, activation='softmax', loss='categorical_crossentropy', optimizer='adam', file_model='model.json'):
        self.input_dim = input_dim
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self._create_model()

    def _create_model(self):
        self.model = Sequential()

        self.model.add(Embedding(self.input_dim, 100, input_length=1000))
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



def main():
    CORPUS = 'data/fine.txt'
    
    #from keras.datasets import reuters
    #(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=1000, test_split=0.2)
    
    tokenizer = Tokenizer(nb_words=1000)

    texts = []
    labels = []
    for i, line in enumerate(open(CORPUS)):
        d, l, t = line.strip().split('\t')
        labels.append(int(l))
        texts.append(t)

    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    sequences = sequence.pad_sequences(sequences, maxlen=1000)
    
    toSplit = int(len(sequences) * 0.1)

    nb_classes = 14

    X_test = sequences[0:toSplit]
    y_test_orig = labels[0:toSplit]
    y_test = np_utils.to_categorical(y_test_orig, nb_classes)

    X_train = sequences[toSplit:]
    y_train = np_utils.to_categorical(labels[toSplit:], nb_classes)

    webClassifier = WebClassifier(X_train.shape[1], nb_classes, epochs=50)
    webClassifier.train(X_train, y_train, validation_split=0.4)
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
