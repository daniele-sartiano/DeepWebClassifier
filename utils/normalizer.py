#!/usr/bin/env python

import sys
from keras.preprocessing.text import Tokenizer, text_to_word_sequence


def main():
    tokenizer = Tokenizer()
    for line in sys.stdin:
        t = line.strip()
        print ' '.join(text_to_word_sequence(t, lower=False))


if __name__ == '__main__':
    main()
