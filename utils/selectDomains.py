#!/usr/bin/env python

import sys
import numpy

numpy.random.seed(7)
TH = 1000000

paths = []

for l in sys.stdin:
    l = l.strip()
    if not l.endswith('.old'):
        paths.append(l)

numpy.random.shuffle(paths)

for l in paths[:TH]:
    print '%s\t-1\t-1' % l
