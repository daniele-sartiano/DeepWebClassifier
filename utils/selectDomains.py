#!/usr/bin/env python

import sys
import numpy

numpy.random.seed(7)
paths = []

for l in sys.stdin:
    l = l.strip()
    if not l.endswith('.old'):
        paths.append(l)

numpy.random.shuffle(paths)

for l in paths:
    print '%s\t-1\t-1' % l
