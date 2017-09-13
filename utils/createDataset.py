#!/usr/bin/env python

import sys
import os
import argparse
import multiprocessing

from bs4 import BeautifulSoup, Comment
import re
 
def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    elif any(x in str(element.encode('utf-8')).lower() for x in ('<script', '[if', '<a', '<link', '<![endif]')):
        return False
    return True


def extractor(path, headings=False):
    try:
        html = open(path).read()
        if not html:
            return ''

        soup = BeautifulSoup(html, "lxml")
        
        decompose_tags = ['script', 'style']
        for tag in decompose_tags:
            for el in soup.findAll(tag):
                if el:
                    #print >> sys.stderr, 'Decomposing', el
                    el.decompose()

        words = []
        if soup.body:
            for tag in soup.body.stripped_strings:
                words.append(' '.join([w.strip() for w in tag.encode('utf-8').split()]))

        data = ' '.join(words)

        if headings:
            titles = []
            for link in soup.findAll(['h1', 'h2']):
                titles.append(' '.join([w.strip().lower() for w in link.text.encode('utf-8').split()]))        
            titles = set(titles)
            return data, ' '.join([el for el in titles])
        
        #data = soup.findAll(text=True)
        #return ' '.join(' '.join([' '.join([e.strip() for e in el.strip().split() if e.strip()]) for el in filter(visible, data) if el.strip()]).splitlines())
        return data

    except Exception as e:
        print >> sys.stderr, 'Exception', e
        return ''


global_path = ''
headings = False

def extract(line):
    domain, _, label = line.strip().split('\t')
    path = os.path.join(global_path, os.path.join(*[c for c in domain][:3]), domain)

    if not path:
        return None
    journal = os.path.join(path, '.journal')

    if os.path.exists(journal):
        text = []
        h = []
        for line in open(journal):
            if not line.strip():
                continue

            row = line.strip().split()[0]
            fname = row[row.find(domain)+len(domain)+1:]
            
            if fname and os.path.exists(os.path.join(path, fname)):
                r = extractor(os.path.join(path, fname), headings)
                if headings:
                    text.append(r[0])
                    h.append(r[1])
                else:
                    text.append(r)

        if headings:
            return '%s\t%s\t%s\t%s' % (domain, '___deep_classifier_project___'.join(text), '___deep_classifier_project___'.join(h), label)
        return '%s\t%s\t%s' % (domain, '___deep_classifier_project___'.join(text), label)
    print >> sys.stderr, 'skipping', path, domain, label
    return None


def main():
    parser = argparse.ArgumentParser(description='Corpus creator')
    parser.add_argument('-d', '--directory', help='the directory with files', type=str, required=True)
    parser.add_argument('-p', '--process', type=int, default=12)
    parser.add_argument('-headings', '--headings', action='store_true')

    args = parser.parse_args()

    print >> sys.stderr, args

    def initialize(args):
        global global_path
        global headings
        global_path = args.directory
        headings = args.headings

    pool = multiprocessing.Pool(args.process, initialize, (args,))

    for r in pool.imap_unordered(extract, sys.stdin, 5):
        if r:
            print r

            
if __name__ == '__main__':
    main()
