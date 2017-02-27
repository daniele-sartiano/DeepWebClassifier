#!/usr/bin/env python

import sys
import gensim
import argparse

class SentenceIterator(object):
    def __init__(self, input=sys.stdin, max=1000000000):
        self.input = input
        self.max = max

    def __iter__(self):
        for i, line in enumerate(self.input):
            yield line.strip().split()
            if i > self.max:
                break


def main():
    parser = argparse.ArgumentParser(description='word2vec')
    parser.add_argument('-s', '--size', help='word embeddings size', type=int, default=300)
    parser.add_argument('-m', '--max-vocab-size', help='max vocab size', type=int, default=10000000)
    parser.add_argument('-mc', '--min-count', help='min count', type=int, default=5)
    parser.add_argument('-w', '--workers', help='# workers', type=int, default=12)

    args = parser.parse_args()

    model = gensim.models.Word2Vec(SentenceIterator(), size=args.size, max_vocab_size=args.max_vocab_size, min_count=args.min_count, workers=args.workers)
    model.init_sims(replace=True)
    model.save('word_embeddings_gensim_%s.txt' % args.size)
    model.save_word2vec_format('word_embeddings_%s' % args.size, , fvocab='vocab_%s' % args.size, binary=False)


if __name__ == '__main__':
    main()

