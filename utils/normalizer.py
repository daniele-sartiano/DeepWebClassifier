#!/usr/bin/env python

import sys
import math
import re
import codecs
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from nltk.tokenize import TweetTokenizer, RegexpTokenizer

import multiprocessing

# sys.path.append('../subword-nmt')
# from apply_bpe import BPE

def windows(sentence, n=3, end='<END>'):
    for token in sentence.split():
        prev = 0
        s = []
        for i in xrange(0, int(math.ceil(len(token)/float(n)))):
            s.append(token[i*n:(i+1)*n])
        s[-1] = '%s%s' % (s[-1], end)
        yield s


def normalize(input=sys.stdin):
    MAX_WORDS=2000

    tokenizer = Tokenizer(nb_words=MAX_WORDS, lower=False)
    tweetTokenizer = TweetTokenizer()

    texts = []
    for line in input:
        sentences = line.strip().split('___deep_classifier_project___')
        for sentence in sentences:        
            sentence = ' '.join(tweetTokenizer.tokenize(sentence)).encode('utf-8')
            s = re.sub(r'\d', '0', sentence)
            texts.append(' '.join([w for w in s.split() if len(w.strip().decode('utf-8')) > 2]))
            if len(texts) % 100000 == 0:
                print >> sys.stderr, 'read %s sentences' % len(texts)
    print >> sys.stderr, 'loading done'
    tokenizer.fit_on_texts(texts)
    return texts


def multi_normalize(line):
    regexpTokenizer = RegexpTokenizer(r'\w+')
    texts = []
    sentences = codecs.decode(line, 'utf-8').strip().split('___deep_classifier_project___')
    
    for sentence in sentences:        
        sentence = ' '.join(regexpTokenizer.tokenize(sentence))
        s = re.sub(r'\d', '0', sentence)
        if g_lower:
            s = s.lower()
        s = ' '.join([w for w in s.split() if len(w.strip()) > 2 and len(set(w.strip())) > 1])
        s = s.encode('utf-8')
        if s:
            texts.append(s)
    return texts
    
def normalize_line(line, lower=False, vocabulary=None, window_size=None, bpe=None):
    regexpTokenizer = RegexpTokenizer(r'\w+')
    
    if bpe:
        bpe_encoder = BPE(open(bpe), '@@', None, None)

    sentences = line.strip().split('___deep_classifier_project___')
    texts = []
    
    import codecs

    for sentence in sentences:
        sentence = ' '.join(regexpTokenizer.tokenize(sentence)).encode('utf-8')
        s = re.sub(r'\d', '0', sentence)
        if lower:
            s = s.lower()
        if vocabulary:
            s = ' '.join([w for w in s.split() if len(w) > 2 and w in vocabulary])
        else:
            s = ' '.join([w for w in s.split() if len(w.strip().decode('utf-8')) > 2 and len(set(w.strip().decode('utf-8'))) > 1])
        if window_size:
            r = ' '.join([' '.join(el) for el in windows(s, window_size)])
            s = r
        if bpe:
            s = bpe_encoder.segment(codecs.decode(s, 'utf-8')).strip().encode('utf-8')
        if s:
            texts.append(s)
    return texts

g_lower = False

def main():

    lower = True if sys.argv[1] == '--lower' else False

    def initialize(lower):
        global g_lower
        g_lower = lower
        
    pool = multiprocessing.Pool(24, initialize, (lower, ))

    for r in pool.imap_unordered(multi_normalize, sys.stdin, 5):
        for sentence in r:
            print sentence
    

if __name__ == '__main__':
    main()
