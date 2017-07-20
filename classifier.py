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
from keras import regularizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix

from reader import TextDomainReader, Reader

class WebClassifier(object):
    def __init__(self, reader=None, input_dim_content=-1, batch_size=32, epochs=20, activation='softmax', loss='categorical_crossentropy', optimizer='adam', file_model='web.model', embeddings_weights_content=None, embeddings_dim_content=50, embeddings_weights_domains=None, embeddings_dim_domains=50, input_dim_domains=-1):
        self.reader = reader

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

        if reader is not None:
            self._create_model()

        
    def _create_model(self): 
        content_input = Input(shape=(self.reader.max_sequence_length_content, ))
        content_embeddings = Embedding(
            input_dim=self.input_dim_content,
            output_dim=self.embeddings_dim_content, 
            weights=[self.embeddings_weights_content],
            input_length=self.reader.max_sequence_length_content,
            trainable=True
        )(content_input)

        content_trainable = Convolution1D(filters=2048, kernel_size=5, padding='same', activation='relu')(content_embeddings)
        content_trainable = GlobalMaxPooling1D()(content_trainable)
        content_trainable = Dropout(0.8)(content_trainable)

        # Convolutional block
        conv_blocks = []
        for sz in (3,4,5,6,7,8,9,10):
            conv = Convolution1D(filters=256,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(content_embeddings)
            
            conv = MaxPooling1D(pool_size=5)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        content_conv_block = Concatenate()(conv_blocks)
        content_conv_block = Dropout(0.8)(content_conv_block)

        content_conv_block = Dense(256, activation='relu')(content_conv_block)
        
        # content_not_trainable = Embedding(
        #     input_dim=self.input_dim_content,
        #     output_dim=self.embeddings_dim_content, 
        #     weights=[self.embeddings_weights_content],
        #     input_length=self.reader.max_sequence_length_content,
        #     trainable=False
        # )(content_input)
        # content_not_trainable = Convolution1D(filters=1024, kernel_size=5, padding='same', activation='relu')(content_not_trainable)
        # content_not_trainable = GlobalMaxPooling1D()(content_not_trainable)
                
        domain_input = Input(shape=(self.reader.max_sequence_length_domains, ))
        domain = Embedding(
            input_dim=self.input_dim_domains,
            output_dim=self.embeddings_dim_domains, 
            weights=[self.embeddings_weights_domains],
            input_length=self.reader.max_sequence_length_domains,
            trainable=True
        )(domain_input)
        
        domain = Flatten()(domain)
    
        # x = keras.layers.concatenate([content_trainable, content_not_trainable, domain])
        x = keras.layers.concatenate([content_trainable, content_conv_block, domain])
        #x1 = keras.layers.average([content_trainable, content_not_trainable])
        #x = keras.layers.concatenate([x, x1])
        # x = Dense(128, activation='relu')(x)
        # x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.5)(x)

        output = Dense(self.reader.nb_classes, activation='softmax')(x)

        optim = keras.optimizers.Adam(lr=0.0001) # default lr=0.001
        self.model = Model(inputs=[content_input, domain_input], outputs=output)
        self.model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
        

    def _create_model_2(self): 
        content_input = Input(shape=(self.reader.max_sequence_length_content, ))
        content = Embedding(
            input_dim=self.input_dim_content,
            output_dim=self.embeddings_dim_content, 
            weights=[self.embeddings_weights_content],
            input_length=self.reader.max_sequence_length_content,
            trainable=True
        )(content_input)
        content = Convolution1D(filters=1024, kernel_size=5, padding='same', activation='relu')(content)
        content = GlobalMaxPooling1D()(content)

        
        content2 = Embedding(
            input_dim=self.input_dim_content,
            output_dim=self.embeddings_dim_content, 
            weights=[self.embeddings_weights_content],
            input_length=self.reader.max_sequence_length_content,
            trainable=False
        )(content_input)
        content2 = Convolution1D(filters=1024, kernel_size=5, padding='same', activation='relu')(content2)
        content2 = Convolution1D(filters=512, kernel_size=5, padding='same', activation='relu')(content2)
        content2 = Convolution1D(filters=256, kernel_size=5, padding='same', activation='relu')(content2)
        content2 = GlobalMaxPooling1D()(content2)
                
        domain_input = Input(shape=(self.reader.max_sequence_length_domains, ))
        domain = Embedding(
            input_dim=self.input_dim_domains,
            output_dim=self.embeddings_dim_domains, 
            weights=[self.embeddings_weights_domains],
            input_length=self.reader.max_sequence_length_domains,
            trainable=True
        )(domain_input)
        domain = Flatten()(domain)

        x = keras.layers.concatenate([content, content2, domain])
        x = Dense(32, activation='relu')(x)
        output = Dense(self.reader.nb_classes, activation='softmax')(x)

        self.model = Model(inputs=[content_input, domain_input], outputs=output)
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        
    def _create_model_1(self):
        content_input = Input(shape=(self.reader.max_sequence_length_content, ))
        content = Embedding(
            input_dim=self.input_dim_content,
            output_dim=self.embeddings_dim_content, 
            weights=[self.embeddings_weights_content],
            input_length=self.reader.max_sequence_length_content,
            trainable=True
        )(content_input)

        content = Dropout(0.5)(content)

        # Convolutional block
        conv_blocks = []
        for sz in (3,4,5,6,7,8,9,10):
            conv = Convolution1D(filters=32,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(content)
            
            conv = MaxPooling1D(pool_size=5)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        content = Concatenate()(conv_blocks)
        content = Dropout(0.8)(content)

        content = Dense(256, activation='relu')(content)
        
        domain_input = Input(shape=(self.reader.max_sequence_length_domains, ))
        domain = Embedding(
            input_dim=self.input_dim_domains,
            output_dim=self.embeddings_dim_domains, 
            weights=[self.embeddings_weights_domains],
            input_length=self.reader.max_sequence_length_domains,
            trainable=True
        )(domain_input)
        domain = Flatten()(domain)
        
        x = keras.layers.concatenate([content, domain])
        x = Dense(32, activation='relu')(x)
        output = Dense(self.reader.nb_classes, activation='softmax')(x)

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
        # content.add(Convolution1D(nb_filter=128, filter_length=5, padding='valid', activation='relu'))
        # content.add(MaxPooling1D(5))

        # content.add(Convolution1D(nb_filter=128, filter_length=5, padding='valid', activation='relu'))
        # content.add(MaxPooling1D(5))

        # content.add(Convolution1D(nb_filter=128, filter_length=5, padding='valid', activation='relu'))
        # content.add(MaxPooling1D(35))

        content.add(Convolution1D(filters=1024, kernel_size=5, padding='same', activation='relu'))
        # content.add(Convolution1D(filters=512, kernel_size=5, padding='same', activation='relu'))
        # content.add(Convolution1D(filters=256, kernel_size=5, padding='same', activation='relu'))
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

        # domain.add(Convolution1D(nb_filter=128, filter_length=3, padding='same', activation='relu'))
        # domain.add(GlobalMaxPooling1D())
        domain.add(Dense(64))

        self.model = Sequential()
        #self.model.add(Concatenate(input_shape=, [content, domain]))
        
        self.model.add(Dense(self.reader.nb_classes, activation='softmax'))

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
                           ModelCheckpoint(self.file_model, monitor='val_loss', verbose=True, save_best_only=True)
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
        #self.model.save(self.file_model)
        reader_file = '%s_reader.pickle' % self.file_model
        Reader.save(reader_file, self.reader)


    def load(self):
        self.model = load_model(self.file_model)
        reader_file = '%s_reader.pickle' % self.file_model
        self.reader = TextDomainReader(**Reader.load(reader_file))

        
    def setModel(self):
        self.model = load_model()
        
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

        self.model.add(Dense(self.reader.nb_classes, activation='softmax'))

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


# def load_vectors(f):
#     embeddings_index = {}
#     embeddings_size = None
#     for line in f:
#         if embeddings_size is None:
#             embeddings_size = int(line.strip().split()[-1])
#             continue
#         values = line.split()
#         word = values[0]
#         coefs = numpy.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#     f.close()

#     return embeddings_index, embeddings_size


def main():
    parser = argparse.ArgumentParser(description='WebClassifier')
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train')
    parser_train.set_defaults(which='train')
    
    common_args = [
        (['-f', '--file-model'], {'help':'file model', 'type':str, 'default':'web.model'}),
        (['-i', '--input'], {'help':'input', 'help': 'input, default standard input', 'type':str, 'default':None})    
    ]

    parser_train.add_argument('-mw', '--max-words-content', help='Max words', type=int, required=True)
    parser_train.add_argument('-msl', '--max-sequence-length-content', help='Max sequence length', type=int, required=True)
    parser_train.add_argument('-mwd', '--max-words-domains', help='Max words domains', type=int)
    parser_train.add_argument('-msld', '--max-sequence-length-domains', help='Max sequence length', type=int, required=True)
    parser_train.add_argument('-e', '--embeddings', help='Embeddings', type=str, required=True)
    parser_train.add_argument('-ed', '--embeddings-domains', help='Embeddings for domain', type=str, required=False)
    parser_train.add_argument('-l', '--lower', action='store_true')
    parser_train.add_argument('-epochs', '--epochs', help='Epochs', type=int, default=200)
    parser_train.add_argument('-batch', '--batch', help='# batch', type=int, default=16)
    parser_train.add_argument('-w', '--window', help='window', type=int, default=None)
    parser_train.add_argument('-bpe', '--bpe', help='bpe file', type=str, default=None)

    for arg in common_args:
        parser_train.add_argument(*arg[0], **arg[1])
    
    parser_test = subparsers.add_parser('test')
    parser_test.set_defaults(which='test')

    for arg in common_args:
        parser_test.add_argument(*arg[0], **arg[1])


    parser_visualize = subparsers.add_parser('visualize')
    parser_visualize.set_defaults(which='visualize')

    for arg in common_args:
        parser_visualize.add_argument(*arg[0], **arg[1])
    
    args = parser.parse_args()
    
    numpy.random.seed(7)

    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', level=logging.INFO)

    input = open(args.input) if args.input else sys.stdin

    if args.which == 'train':
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

            reader = TextDomainReader(input=input, 
                                      max_sequence_length_content=args.max_sequence_length_content, 
                                      max_words_content=args.max_words_content, 
                                      max_sequence_length_domains=args.max_sequence_length_domains, 
                                      max_words_domains=args.max_words_domains,
                                      domains_vocabulary=vocabulary,
                                      lower=args.lower,
                                      logger=logging,
                                      window=args.window,
                                      bpe=args.bpe)

        X_train, y_train, X_dev, y_dev, y_dev_orig = reader.read()
        logging.info('X_train %s %s - y_train %s' % (len(X_train[0]), len(X_train[1]), len(y_train)))

        logging.info('Reading Embedings: using the file %s, max words content %s, max sequence length %s content' % (args.embeddings, args.max_words_content, args.max_sequence_length_content))

        num_words_content, embedding_matrix_content, embeddings_size_content = Reader.read_embeddings(args.embeddings,
                                                                                    reader.max_words_content,
                                                                                    reader.tokenizer_content.word_index, args.lower)

        logging.info('Reading Embedings: using the file %s, max words domains %s, max sequence length %s domains' % (args.embeddings_domains, args.max_words_domains, args.max_sequence_length_domains))


        num_words_domains, embedding_matrix_domains, embeddings_size_domains = Reader.read_embeddings(args.embeddings_domains,
                                                                                    reader.max_words_domains,
                                                                                    reader.tokenizer_domains.word_index, args.lower)
    
        webClassifier = WebClassifier(
            reader=reader,
            file_model=args.file_model,
            input_dim_content=num_words_content+1,
            embeddings_dim_content=embeddings_size_content, 
            embeddings_weights_content=embedding_matrix_content, 
            embeddings_dim_domains=embeddings_size_domains, 
            embeddings_weights_domains=embedding_matrix_domains, 
            input_dim_domains=num_words_domains+1,
            epochs=args.epochs, 
            batch_size=args.batch)

        logging.info(webClassifier.describeModel())

        webClassifier.train(X_train, y_train, validation_split=0.3)
        webClassifier.save()

        webClassifier.load()
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

    elif args.which == 'test':
        webClassifier = WebClassifier(file_model=args.file_model)
        webClassifier.load()
        X, y, y_orig = webClassifier.reader.read_for_test()
        webClassifier.evaluate(X, y)
        y_pred, p = webClassifier.predict(X)
        webClassifier.modelSummary()
        logging.info('\n%s' % classification_report(y_orig, y_pred))
        logging.info('\n%s' % confusion_matrix(y_orig, y_pred))

    elif args.which == 'visualize':
        webClassifier = WebClassifier(file_model=args.file_model)
        webClassifier.load()
        from keras.utils import plot_model
        plot_model(webClassifier.model, to_file='%s.png' % args.file_model)

if __name__ == '__main__':
    main()
