#!/usr/bin/env python


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
    model = gensim.models.Word2Vec(SentenceIterator(), size=300, max_vocab_size=10000000,min_count=5, workers=20)                                                                                                                           
    model.init_sims(replace=True)                                                                                                                                                                         
    model.save('w2v.txt')   
    model.save_word2vec_format('', fvocab=None, binary=False)
