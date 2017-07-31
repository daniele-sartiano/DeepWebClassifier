#!/usr/bin/env python

import sys
import gensim
import argparse

import logging

class SentenceIterator(object):
    def __init__(self, input=sys.stdin):
        self.input = input
        self.all = input.readlines()

    def __iter__(self):
        for i, line in enumerate(self.all):
            if i % 10000 == 0:
                print('%s lines' % i, file=sys.stderr)
            if not line.strip():
                continue
            yield line.strip().split()


def main():
    parser = argparse.ArgumentParser(description='word2vec')
    parser.add_argument('-f', '--filename', help='word embeddings filename', type=str, required=True)
    parser.add_argument('-s', '--size', help='word embeddings size', type=int, default=300)
    parser.add_argument('-m', '--max-vocab-size', help='max vocab size', type=int, default=1000000)
    parser.add_argument('-mc', '--min-count', help='min count', type=int, default=5)
    parser.add_argument('-wi', '--window', help='window', type=int, default=5)
    parser.add_argument('-w', '--workers', help='# workers', type=int, default=18)
    parser.add_argument('-l', '--load', help='# load', action='store_true')

    args = parser.parse_args()
    
    logger = logging.getLogger('w2v.py')

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if args.load:
        model = gensim.models.Word2Vec.load('data/word_embeddings_gensim_%s.bin' % args.size)
        model.train(SentenceIterator())
    else:
        model = gensim.models.word2vec.Word2Vec(SentenceIterator(), size=args.size, max_vocab_size=args.max_vocab_size, min_count=args.min_count, workers=args.workers)
    
    model.save('data/word_embeddings_gensim_%s.bin' % args.size)
    model.wv.save_word2vec_format(args.filename, binary=False)


if __name__ == '__main__':
    main()

