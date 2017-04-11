from utils.normalizer import normalize_line

import cPickle
import numpy

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils

from copy_reg import pickle
from types import MethodType


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
        
class TextDomainReader(Reader):
    def __init__(self, input,
                 max_sequence_length_content, max_words_content, 
                 max_sequence_length_domains, max_words_domains, domains_vocabulary):
        super(TextDomainReader, self).__init__(input)
        self.max_sequence_length_content = max_sequence_length_content
        self.max_words_content = max_words_content
        self.max_sequence_length_domains = max_sequence_length_domains
        self.max_words_domains = max_words_domains
        self.splitter = Splitter(domains_vocabulary)
        self.nb_classes = -1
        self.tokenizer_content = None
        self.tokenizer_domains = None

        self.fields += [
            'max_sequence_length_content',
            'max_words_content',
            'max_sequence_length_domains',
            'max_words_domains',
            'splitter',
            'nb_classes',
            'tokenizer_content',
            'tokenizer_domains'
        ]
        
        
    def read(self):
        labels = []
        texts = []
        domains = []
        
        for line in self.input:
            d, t, l = line.strip().split('\t')
            for label in l.split(','):
                l = 0 if int(label) == 13 else int(label)
                labels.append(l)
                texts.append(' '.join(normalize_line(t)))

                d_words = sorted([el for el in self.splitter.split(d[:-3])], key=lambda x:x[1], reverse=True)
                selected = sorted(set([tok for words, th in d_words[:3] for tok in words if len(tok.decode('utf8')) > 2]), key=lambda x: len(x), reverse=True)
                domains.append(' '.join(selected) if len(selected) > 0 else d[:-3])

        self.nb_classes = len(set(labels))
        
        self.tokenizer_domains = Tokenizer(nb_words=self.max_words_domains, lower=False)
        self.tokenizer_domains.fit_on_texts(domains)
        sequences_domains = self.tokenizer_domains.texts_to_sequences(domains)
        sequences_domains = sequence.pad_sequences(sequences_domains, maxlen=self.max_sequence_length_domains)

        self.tokenizer_content = Tokenizer(nb_words=self.max_words_content, lower=False)
        self.tokenizer_content.fit_on_texts(texts)
        sequences_content = self.tokenizer_content.texts_to_sequences(texts)
        sequences_content = sequence.pad_sequences(sequences_content, maxlen=self.max_sequence_length_content)

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
