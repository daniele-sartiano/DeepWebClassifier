#!/usr/bin/env python

import sys
import re
from keras.preprocessing.text import Tokenizer, text_to_word_sequence


def normalize(input=sys.stdin, MAX_WORDS=20000):
    tokenizer = Tokenizer(nb_words=MAX_WORDS, lower=False)

    texts = []
    for line in input:
        sentences = line.strip().split('___deep_classifier_project___')
        for sentence in sentences:
            s = re.sub(r'\d', '0', ' '.join(text_to_word_sequence(sentence, lower=False)))
            texts.append(' '.join([w for w in s.split() if len(w) > 2]))
            if len(texts) % 100000 == 0:
                print >> sys.stderr, 'read %s sentences' % len(texts)
    print >> sys.stderr, 'loading done'

    tokenizer.fit_on_texts(texts)
    return texts

    
def normalize_line(line, MAX_WORDS=20000):
    tokenizer = Tokenizer(nb_words=MAX_WORDS, lower=False)
    sentences = line.strip().split('___deep_classifier_project___')
    texts = []
    for sentence in sentences:
        s = re.sub(r'\d', '0', ' '.join(text_to_word_sequence(sentence, lower=False)))
        texts.append(' '.join([w for w in s.split() if len(w) > 2]))
    tokenizer.fit_on_texts(texts)
    return texts

def main():
    sentences = normalize()
    for sentence in sentences:
        print sentence

if __name__ == '__main__':
    main()
