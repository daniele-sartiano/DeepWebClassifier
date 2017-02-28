#!/usr/bin/env python

import sys
import re
from keras.preprocessing.text import Tokenizer, text_to_word_sequence


def normalize(input=sys.stdin):
    for line in input:
        t = line.strip()
        yield re.sub(r'\d', '0', ' '.join(text_to_word_sequence(t, lower=False)))
    
def normalize_line(line):
    return re.sub(r'\d', '0', ' '.join(text_to_word_sequence(line, lower=False)))

def main():
    for t in normalize():
        print t

if __name__ == '__main__':
    main()
