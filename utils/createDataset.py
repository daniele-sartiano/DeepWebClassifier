#!/usr/bin/env python

import sys
import os
import argparse
import multiprocessing

from bs4 import BeautifulSoup
import re
 
def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    elif any(x in str(element.encode('utf-8')).lower() for x in ('<script', '[if', '<a', '<link', '<![endif]')):
        return False
    return True


def extractor(path):
    html = open(path).read()
    if not html:
        return ''

    soup = BeautifulSoup(html, "html5lib")
    data = soup.findAll(text=True)

    return ' '.join(' '.join([' '.join([e.strip() for e in el.strip().split() if e.strip()]) for el in filter(visible, data) if el.strip()]).splitlines())


global_path = ''

def extract(line):
    domain, _, label = line.strip().split('\t')
    path = os.path.join(global_path, os.path.join(*[c for c in domain][:3]), domain)

    if not path:
        return None
    journal = os.path.join(path, '.journal')

    if os.path.exists(journal):
        text = []
        for line in open(journal):
            if not line.strip():
                continue

            row = line.strip().split()[0]
            fname = row[row.find(domain)+len(domain)+1:]
            
            if fname:
                text.append(extractor(os.path.join(path, fname)).encode('utf-8'))

        return '%s\t%s\t%s' % (domain, '___deep_classifier_project___'.join(text), label)
    print >> sys.stderr, 'skipping', domain, label
    return None


def main():
    parser = argparse.ArgumentParser(description='Corpus creator')
    parser.add_argument('-d', '--directory', help='the directory with files', type=str, required=True)
    parser.add_argument('-p', '--process', type=int, default=12)
    
    args = parser.parse_args()

    def initialize(args):
        global global_path
        global_path = args.directory

    pool = multiprocessing.Pool(args.process, initialize, (args,))

    for r in pool.imap_unordered(extract, sys.stdin, 5):
        if r:
            print r

            
if __name__ == '__main__':
    main()
