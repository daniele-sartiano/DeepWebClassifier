from utils.normalizer import normalize_line

import sys
import cPickle
import numpy
import logging

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils

from copy_reg import pickle
from types import MethodType


class Splitter(object):
    MIN_LEN = 2
    
    OTHER = ['di', 'a', 'da', 'in', 'su', 'il', 'lo', 'la', 'un', 'e', 'i', 'o', 'al', 'd', 'l', 'c']

    def __init__(self, resource):
        self.vocabulary = set()
        for token in resource:
            self.vocabulary.add(token.strip().lower())
        for w in self.OTHER:
            self.vocabulary.add(w)

 
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
                split = tokens.split()
                yield split, sum([len(tok)/float(len(tokens)) for tok in split])

    
    def inVocabulary(self, word):
        return (word in self.vocabulary and word not in self.OTHER)


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


# reader = ParallelReader(sys.stdin, 18, splitter)

# texts = []
# labels = []
# domains = []
    
# for domain, label, content, domain_words in reader.read():
#     labels.append(label)
#     texts.append(content)
#     domains.append(domain_words)


class ParallelReader(object):
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


class Reader(object):
    def __init__(self, input):
        self.input = input
        self.fields = []

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)
        
    @staticmethod
    def save(f, obj):
        cPickle.dump({el: getattr(obj, el) for el in obj.fields}, open(f, 'wb'))

    @staticmethod
    def load(f):
        return cPickle.load(open(f))
        

    @staticmethod
    def load_vectors(f, lower=True):
        embeddings_index = {}
        embeddings_size = None
        for line in f:
            if embeddings_size is None:
                embeddings_size = int(line.strip().split()[-1])
                continue
            values = line.split()
            word = values[0].lower() if lower else value[0]
            coefs = numpy.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        return embeddings_index, embeddings_size

    @staticmethod
    def read_embeddings(f, max_words, word_index, lower=True):
        # embeddings matrix
        embeddings_index, embeddings_size = Reader.load_vectors(open(f), lower)
        print >> sys.stderr, '*'*80
        print >> sys.stderr, len(embeddings_index)
        num_words = min(max_words, len(word_index))
        embedding_matrix = numpy.zeros((num_words + 1, embeddings_size))
        unk = []
        for word, i in word_index.items():
            if i > max_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                unk.append(word)
        print >> sys.stderr, 'unk words', len(unk)
        print >> sys.stderr, unk
        print >> sys.stderr, '*'*80
        return num_words, embedding_matrix, embeddings_size

    @staticmethod
    def discard_zeros(X, embeddings):
        debug_counter = 0
        X_post = []
        for example in X:
            ex = []
            for index in example:
                if not embeddings[index].any():
                    debug_counter += 1
                    continue
                ex.append(embeddings[index])
            X_post.append(ex)
        print >> sys.stderr, 'discard zeros: ', debug_counter
        return numpy.asarray(X_post)


class TextDomainReader(Reader):
    def __init__(self, input=None,
                 max_sequence_length_content=None, max_words_content=None, 
                 max_sequence_length_domains=None, max_words_domains=None, 
                 content_vocabulary=None, domains_vocabulary=None,
                 tokenizer_content=None, tokenizer_domains=None,
                 nb_classes=-1, lower=False, logger=None, bpe=None, window=None):

        super(TextDomainReader, self).__init__(input)

        if input is None:
            self.input = sys.stdin
        self.max_sequence_length_content = max_sequence_length_content
        self.max_words_content = max_words_content
        self.max_sequence_length_domains = max_sequence_length_domains
        self.max_words_domains = max_words_domains
        self.content_vocabulary = content_vocabulary
        self.domains_vocabulary = domains_vocabulary
        self.splitter = Splitter(self.domains_vocabulary)
        self.nb_classes = nb_classes
        self.tokenizer_content = tokenizer_content
        self.tokenizer_domains = tokenizer_domains
        self.lower = lower
        self.window = window
        self.bpe = bpe
        
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', level=logging.INFO)
            self.logger = logging
            
        self.fields += [
            'max_sequence_length_content',
            'max_words_content',
            'max_sequence_length_domains',
            'max_words_domains',
            'content_vocabulary',
            'domains_vocabulary',
            'nb_classes',
            'tokenizer_content',
            'tokenizer_domains',
            'lower',
            'window',
            'bpe'
        ]


    @staticmethod
    def extract_windows(sequence, window_size):
        d = []
        for i in range(len(sequence)-window_size+1):
            d.append(sequence[i:i+window_size])
        return d


    def _read(self):
        labels = []
        texts = []
        domains = []

        self.logger.info('Reading corpus')
        
        for i, line in enumerate(self.input):
            d, t, l = line.strip().split('\t')
            for label in l.split(','):
                l = 0 if int(label) == 13 else int(label)
                labels.append(l)
                            
                if self.window:
                    texts.append(' '.join(normalize_line(t, lower=self.lower, window=self.window)))
                elif self.bpe:
                    texts.append(' '.join(normalize_line(t, lower=self.lower, bpe=self.bpe)))
                else:
                    texts.append(' '.join(normalize_line(t, lower=self.lower, vocabulary=self.content_vocabulary)))

                d_words = sorted([el for el in self.splitter.split(d[:-3])], key=lambda x:x[1], reverse=True)
                selected = sorted(set([tok for words, th in d_words[:3] for tok in words if len(tok.decode('utf8')) > 2 and self.splitter.inVocabulary(tok)]), key=lambda x: len(x), reverse=True)
                domains.append(' '.join(selected) if len(selected) > 0 else d[:-3])
        return labels, texts, domains
        
        
    def read_for_test(self):
        labels, texts, domains = self._read()

        sequences_domains = self.tokenizer_domains.texts_to_sequences(domains)
        sequences_domains = sequence.pad_sequences(sequences_domains, padding='post', truncating='post', maxlen=self.max_sequence_length_domains)

        sequences_content = self.tokenizer_content.texts_to_sequences(texts)
        sequences_content = sequence.pad_sequences(sequences_content, padding='post', truncating='post', maxlen=self.max_sequence_length_content)
        
        y_orig = labels
        y = np_utils.to_categorical(y_orig, self.nb_classes)
        
        X = sequences_content
        X_domains = sequences_domains
        
        return [X, X_domains], y, y_orig

        
    def read(self, split=True):
        labels, texts, domains = self._read()
        self.nb_classes = 12 #len(set(labels)) #43

        self.logger.info('collecting domains sequences')
        
        self.tokenizer_domains = Tokenizer(num_words=self.max_words_domains, lower=False)
        self.tokenizer_domains.fit_on_texts(domains)
        sequences_domains = self.tokenizer_domains.texts_to_sequences(domains)
        sequences_domains = sequence.pad_sequences(sequences_domains, padding='post', truncating='post', maxlen=self.max_sequence_length_domains)

        self.logger.info('collecting content sequences')

        self.tokenizer_content = Tokenizer(num_words=self.max_words_content, lower=False)
        self.tokenizer_content.fit_on_texts(texts)
        sequences_content = self.tokenizer_content.texts_to_sequences(texts)
        sequences_content = sequence.pad_sequences(sequences_content, padding='post', truncating='post', maxlen=self.max_sequence_length_content)
        #sequences_content = numpy.asarray([self.extract_windows(seq, 5) for seq in sequences_content])
        
        self.logger.info('Splitting corpus')

        if split:
            rng_state = numpy.random.get_state()
            numpy.random.shuffle(sequences_content)
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(labels)
            numpy.random.set_state(rng_state)
            numpy.random.shuffle(sequences_domains)

            toSplit = int(len(sequences_content) * 0.1)
            X_dev = sequences_content[0:toSplit]
            X_dev_domains = sequences_domains[0:toSplit]

            y_dev_orig = labels[0:toSplit]
            y_dev = np_utils.to_categorical(y_dev_orig, self.nb_classes)
            
            X_train = sequences_content[toSplit:]
            X_train_domains = sequences_domains[toSplit:]
            
            y_train = np_utils.to_categorical(labels[toSplit:], self.nb_classes)

            return [X_train, X_train_domains], y_train, [X_dev, X_dev_domains], y_dev, y_dev_orig
        else:
            return [sequences_content, sequences_domains], np_utils.to_categorical(labels, self.nb_classes)
            
