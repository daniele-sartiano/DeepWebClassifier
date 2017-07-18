#!/usr/bin/env python

import sys
import gensim
import argparse

class SentenceIterator(object):
    def __init__(self, input=sys.stdin):
        self.input = input

    def __iter__(self):
        for i, line in enumerate(self.input):
            if i % 10000 == 0:
                print >> sys.stderr, '%s lines' % i
            yield line.strip().split()


def main():
    parser = argparse.ArgumentParser(description='word2vec')
    parser.add_argument('-f', '--filename', help='word embeddings filename', type=str, required=True)
    parser.add_argument('-s', '--size', help='word embeddings size', type=int, default=300)
    parser.add_argument('-m', '--max-vocab-size', help='max vocab size', type=int, default=1000000)
    parser.add_argument('-mc', '--min-count', help='min count', type=int, default=5)
    parser.add_argument('-wi', '--window', help='window', type=int, default=5)
    parser.add_argument('-w', '--workers', help='# workers', type=int, default=18)

    args = parser.parse_args()

    model = gensim.models.word2vec.Word2Vec(SentenceIterator(), window=args.window, size=args.size, max_vocab_size=args.max_vocab_size, min_count=args.min_count, workers=args.workers)

    model.save('data/word_embeddings_gensim_%s.bin' % args.size)
    model.wv.save_word2vec_format(args.filename, binary=False)


if __name__ == '__main__':
    main()

