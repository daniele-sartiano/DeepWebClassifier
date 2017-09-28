#!/usr/bin/env python
import sys
import math

def windows(sentence, n=3, end='<END>'):
    for token in sentence.split():
        prev = 0
        s = []
        for i in xrange(0, int(math.ceil(len(token)/float(n)))):
            s.append(token[i*n:(i+1)*n])
        s[-1] = '%s%s' % (s[-1], end)
        yield s
                

def main():
    for sentence in sys.stdin:
        for el in windows(sentence):
            print ' '.join(el),
        print

if __name__ == '__main__':
    main()
