#!/usr/bin/env python

import sys
import re
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from nltk.tokenize import TweetTokenizer

import multiprocessing


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


def multi_normalize(line, lower=False):
    tweetTokenizer = TweetTokenizer()
    texts = []
    sentences = line.strip().split('___deep_classifier_project___')

    for sentence in sentences:        
        sentence = ' '.join(tweetTokenizer.tokenize(sentence)).encode('utf-8')
        s = re.sub(r'\d', '0', sentence)
        s = ' '.join([w for w in s.split() if len(w.strip().decode('utf-8')) > 2])
        if lower:
            s.lower()
        texts.append(s)
    return texts
    
def normalize_line(line, lower=False):
    tweetTokenizer = TweetTokenizer()

    sentences = line.strip().split('___deep_classifier_project___')
    texts = []
    for sentence in sentences:
        sentence = ' '.join(tweetTokenizer.tokenize(sentence)).encode('utf-8')
        s = re.sub(r'\d', '0', sentence)
        texts.append(' '.join([w for w in s.split() if len(w) > 2]))
    #tokenizer.fit_on_texts(texts)
    return texts

def main():
    pool = multiprocessing.Pool(18)

    for r in pool.imap_unordered(multi_normalize, sys.stdin, 5):
        for sentence in r:
            print sentence
    

if __name__ == '__main__':
    main()
