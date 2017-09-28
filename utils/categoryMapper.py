#!/usr/bin/env python

import sys

TOURISM_SET = [
    "1",
    "19", 
    "20", 
    "21", 
    "22", 
    "23", 
    "24", 
    "25", 
    "26", 
    "30", 
    "42"
    ]

TOURISM_MAP = {"7": "21", "14": "23"}

def main():
    for line in sys.stdin:
        r = line.strip().split('\t')
        labels = r[-1].split(',')
        for label in labels:
            if label in TOURISM_SET:
                index = TOURISM_SET.index(label)
            elif label in TOURISM_MAP:
                index = TOURISM_SET.index(TOURISM_MAP[label])
            else:
                index = TOURISM_SET.index("1")
            r[-1] = index
            print '\t'.join([str(el) for el in r])

if __name__ == '__main__':
    main()
