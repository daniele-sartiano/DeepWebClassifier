#!/usr/bin/env python

import sys
import os
import argparse
import logging
import multiprocessing

import pandas
import numpy
import math
import keras
from keras.layers import Dense, Activation, LSTM, SimpleRNN, GRU, Dropout, Input, Flatten, GlobalMaxPooling1D
from keras.layers.merge import Concatenate
from keras.layers.embeddings import Embedding
#from keras.layers.merge import Concatenate
from keras.models import Sequential, Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix

from reader import TextDomainReader, Reader

class WebClassifier(object):
    def __init__(self, reader, input_dim_content=-1, batch_size=32, epochs=20, activation='softmax', loss='categorical_crossentropy', optimizer='adam', file_model='model.json', embeddings_weights_content=None, embeddings_dim_content=50, embeddings_weights_domains=None, embeddings_dim_domains=50, input_dim_domains=-1):
        self.reader = reader
        self.nb_classes = reader.nb_classes

        self.input_dim_content = input_dim_content
        self.input_dim_domains = input_dim_domains
        self.batch_size = batch_size
        self.epochs = epochs
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        
        self.embeddings_weights_content = embeddings_weights_content
        self.embeddings_dim_content = embeddings_dim_content
        self.embeddings_weights_domains = embeddings_weights_domains
        self.embeddings_dim_domains = embeddings_dim_domains

        self.file_model = file_model
        self._create_model()

    def _create_model(self): 
        content_input = Input(shape=(self.reader.max_sequence_length_content, ))
        content = Embedding(
            input_dim=self.input_dim_content,
            output_dim=self.embeddings_dim_content, 
            weights=[self.embeddings_weights_content],
            input_length=self.reader.max_sequence_length_content,
            trainable=True
        )(content_input)
        content = Convolution1D(filters=1024, kernel_size=5, border_mode='same', activation='relu')(content)
        content = GlobalMaxPooling1D()(content)
        #content = Dense(256, activation='relu')(content)
        
        domain_input = Input(shape=(self.reader.max_sequence_length_domains, ))
        domain = Embedding(
            input_dim=self.input_dim_domains,
            output_dim=self.embeddings_dim_domains, 
            weights=[self.embeddings_weights_domains],
            input_length=self.reader.max_sequence_length_domains,
            trainable=True
        )(domain_input)
        domain = Flatten()(domain)
        # domain = Dense(256, activation='relu')(domain)

        #domain = Convolution1D(filters=128, kernel_size=3, border_mode='same', activation='relu')(domain)
        #domain = GlobalMaxPooling1D()(domain)
        
        x = keras.layers.concatenate([content, domain])
        x = Dense(32, activation='relu')(x)
        output = Dense(self.nb_classes, activation='softmax')(x)

        self.model = Model(inputs=[content_input, domain_input], outputs=output)
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        

    def _create_model_sequential(self):
        content = Sequential()
        content.add(Embedding(
            self.input_dim_content, 
            self.embeddings_dim_content, 
            weights=[self.embeddings_weights_content],
            input_length=self.reader.max_sequence_length_content,
            trainable=False
        ))

        content.add(Dropout(0.2))
        # content.add(Convolution1D(nb_filter=128, filter_length=5, border_mode='valid', activation='relu'))
        # content.add(MaxPooling1D(5))

        # content.add(Convolution1D(nb_filter=128, filter_length=5, border_mode='valid', activation='relu'))
        # content.add(MaxPooling1D(5))

        # content.add(Convolution1D(nb_filter=128, filter_length=5, border_mode='valid', activation='relu'))
        # content.add(MaxPooling1D(35))

        content.add(Convolution1D(filters=1024, kernel_size=5, border_mode='same', activation='relu'))
        content.add(GlobalMaxPooling1D())

        #content.add(Flatten())
        content.add(Dense(256, activation='relu'))

        #content.add(Dense(self.nb_classes, activation='softmax'))

        domain = Sequential()

        domain.add(Embedding(
            self.input_dim_domains,
            self.embeddings_dim_domains, 
            weights=[self.embeddings_weights_domains],
            input_length=self.reader.max_sequence_length_domains,
            trainable=False
        ))

        domain.add(Flatten())

        # domain.add(Convolution1D(nb_filter=128, filter_length=3, border_mode='same', activation='relu'))
        # domain.add(GlobalMaxPooling1D())
        domain.add(Dense(64))

        self.model = Sequential()
        #self.model.add(Concatenate(input_shape=, [content, domain]))
        
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
        y = self.model.predict(X_test).argmax(axis=-1)
        print y
        #p = self.model.predict_proba(X_test)
        p = []
        return y, p


    def save(self):
        self.model.save(self.file_model)
        reader_file = '%s_reader.pickle' % self.file_model
        Reader.save(reader_file, self.reader)
        # serialize
        # model_json = self.model.to_json()
        # with open(self.file_model, "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # self.model.save_weights("model.h5")
        # print("Saved model to disk")


    def load(self):
        self.model = load_model(self.file_model)
        reader_file = '%s_reader.pickle' % self.file_model
        self.reader = TextDomainReader(**Reader.load(reader_file))
        
    def describeModel(self):
        self.model.summary()
        return 'embeddings content size %s, embeddings domains size %s, epochs %s\n%s' % (self.embeddings_dim_domains, self.embeddings_dim_content, self.epochs, self.model.to_yaml())

    def modelSummary(self):
        self.model.summary()



class WebClassifierMLP(WebClassifier):


    def _create_model(self):
        self.model = Sequential()
        self.model.add(Embedding(
            self.input_dim, 
            self.embeddings_dim, 
            weights=[self.embeddings_weights],
            input_length=self.reader.max_sequence_length_content,
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
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')


    common_args = [
        (['-f', '--file-model'], {'help':'file model', 'type':str, 'default':'model'}),
        (['-i', '--input'], {'help':'input', 'help': 'input, default standard input', 'type':str, 'default':None}),
        
        ]

    parser_train.add_argument('-mw', '--max-words-content', help='Max words', type=int, required=True)
    parser_train.add_argument('-msl', '--max-sequence-length-content', help='Max sequence length', type=int, required=True)
    parser_train.add_argument('-mwd', '--max-words-domains', help='Max words domains', type=int)
    parser_train.add_argument('-msld', '--max-sequence-length-domains', help='Max sequence length', type=int, required=True)
    parser_train.add_argument('-e', '--embeddings', help='Embeddings', type=str, required=True)
    parser_train.add_argument('-ed', '--embeddings-domains', help='Embeddings for domain', type=str, required=False)
    parser_train.add_argument('-epochs', '--epochs', help='Epochs', type=int, default=200)
    parser_train.add_argument('-batch', '--batch', help='# batch', type=int, default=16)

    for arg in common_args:
        parser_train.add_argument(*arg[0], **arg[1])
    
    parser_test = subparsers.add_parser('test')
    parser_test.set_defaults(which='test')

    #parser_test.add_argument('', '--batch', help='# batch', type=int, default=16)
    
    args = parser.parse_args()
    
    numpy.random.seed(7)

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', level=logging.INFO)

    input = open(args.input) if args.input else sys.stdin

    logging.info('Reading vocabulary')
    
    reader = None
    if args.embeddings_domains:
        vocabulary = None
        for w in open(args.embeddings_domains):
            # skipping first line
            if vocabulary is None:
                vocabulary = set()
                continue
            vocabulary.add(w.split(' ')[0].strip().lower())
        for w in ['di', 'a', 'da', 'in', 'su', 'il', 'lo', 'la', 'un', 'e', 'i', 'o', 'al', 'd', 'l', 'c']:
            vocabulary.add(w)

        reader = TextDomainReader(input, args.max_sequence_length_content, args.max_words_content, 
                                  args.max_sequence_length_domains, args.max_words_domains, vocabulary, logging)


    X_train, y_train, X_dev, y_dev, y_dev_orig = reader.read()
    
    logging.info('Reading Embedings: using the file %s, max words content %s, max sequence length %s content' % (args.embeddings, args.max_words_content, args.max_sequence_length_content))

    # prepare embedding matrix
    embeddings_index, embeddings_size = load_vectors(open(args.embeddings))

    nb_words = min(reader.max_words_content, len(reader.tokenizer_content.word_index))
    embedding_matrix = numpy.zeros((nb_words + 1, embeddings_size))
    for word, i in reader.tokenizer_content.word_index.items():
        if i > reader.max_words_content:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    logging.info('Reading Embedings: using the file %s, max words domains %s, max sequence length %s domains' % (args.embeddings_domains, args.max_words_domains, args.max_sequence_length_domains))

    # embeddings matrix for domains
    embeddings_index_domains, embeddings_size_domains = load_vectors(open(args.embeddings_domains))

    nb_words_domains = min(reader.max_words_domains, len(reader.tokenizer_domains.word_index))
    embedding_matrix_domains = numpy.zeros((nb_words_domains + 1, embeddings_size_domains))
    for word, i in reader.tokenizer_domains.word_index.items():
        if i > reader.max_words_domains:
            continue
        embedding_vector = embeddings_index_domains.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix_domains[i] = embedding_vector

    embeddings_index = None
    embeddings_index_domains = None

    webClassifier = WebClassifier(
        reader, 
        input_dim_content=nb_words+1, 
        embeddings_dim_content=embeddings_size, 
        embeddings_weights_content=embedding_matrix, 
        embeddings_dim_domains=embeddings_size_domains, 
        embeddings_weights_domains=embedding_matrix_domains, 
        input_dim_domains=nb_words_domains+1,
        epochs=args.epochs, 
        batch_size=args.batch)
    
    logging.info(webClassifier.describeModel())

    #X_train = [X_train, X_train_domains]

    webClassifier.train(X_train, y_train, validation_split=0.3)
    webClassifier.save()

    logging.info('Evalutaing the model')

    #X_dev = [X_dev, X_dev_domains]
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
