from __future__ import print_function
from collections import defaultdict
import numpy as np
import cPickle as pickle
import time
import os
import csv
import re
from itertools import groupby
import nltk
from dateutil import parser as dateparser


# regexs
RE_NONALNUM = re.compile(r'\W+')
RE_NONANS = re.compile(r'[^\w\s]+')
RE_DIGIT = re.compile(r'\d+')
RE_URL = re.compile(r'https?://')
RE_NONWORD = re.compile(r'[A-Z\d]+')

# labels from 0 to 5, 0 for undefined
all_status = ['not a real question', # 1
              'not constructive',    # 2
              'off topic',           # 3
              'open',                # 4
              'too localized']       # 5
status_map_label = dict((k, int(i + 1)) for i, k in enumerate(all_status))


def gen_datum_feature_dict(datum):
    def norm(string):
        return RE_NONANS.sub('', string).lower()

    def norm_tag(string):
        return RE_NONALNUM.sub('', string).lower()

    def ratio(x, y):
        if y != 0:
            return x / float(y)
        else:
            return 0

    # feature container
    f_dict = defaultdict(dict)
    
    # get text features
    body = datum['BodyMarkdown']
    lines = body.splitlines()
    code = []
    text = []
    sents = []
    title = datum['Title']
    tags = [norm_tag(datum["Tag%d" % i])
            for i in range(1, 6) if datum["Tag%d" % i]]

    # divide post into code and text blocks
    for is_code, group in groupby(lines, lambda l: l.startswith('    ')):
        (code if is_code else text).append('\n'.join(group))

    # build text f_dict features
    f_dict['num']['sent'] = 0
    f_dict['num']['question'] = 0
    f_dict['num']['exclam'] = 0
    f_dict['num']['period'] = 0
    f_dict['num']['initcap'] = 0
    f_dict['num']['istart'] = 0
    f_dict['num']['url'] = 0
    f_dict['num']['digit'] = 0
    f_dict['num']['nonword'] = 0

    for t in text:
        for sent in nltk.sent_tokenize(t):
            f_dict['num']['sent'] += 1
            ss = sent.strip()
            if ss:
                if ss.endswith('?'):
                    f_dict['num']['question'] += 1
                if ss.endswith('!'):
                    f_dict['num']['exclam'] += 1
                if ss.endswith('.'):
                    f_dict['num']['period'] += 1
                if ss.startswith('I '):
                    f_dict['num']['istart'] += 1
                if ss[0].isupper():
                    f_dict['num']['initcap'] += 1

            words = nltk.word_tokenize(norm(sent))
            sents.append(ss)

        f_dict['num']['digit'] += len(RE_DIGIT.findall(t))
        f_dict['num']['url'] += len(RE_URL.findall(t))
        f_dict['num']['nonword'] += len(RE_NONWORD.findall(t))

    f_dict['num']['finalthanks'] = 1 if text and 'thank' in text[-1].lower() else 0
    f_dict['num']['codeblock'] = len(code)
    f_dict['num']['textblock'] = len(text)
    f_dict['num']['lines'] = len(lines)
    f_dict['num']['tags'] = len(tags)
    
    # len features
    f_dict['len']['title'] = len(title)
    f_dict['len']['text'] = sum(len(t) for t in text)
    f_dict['len']['code'] = sum(len(c) for c in code)
    f_dict['len']['firsttext'] = len(text[0]) if text else 0
    f_dict['len']['firstcode'] = len(code[0]) if code else 0
    f_dict['len']['lasttext'] = len(text[-1]) if text else 0
    f_dict['len']['lastcode'] = len(code[-1]) if code else 0
    
    # ratio features
    f_dict['ratio']['tc'] = ratio(f_dict['len']['text'],
                                  f_dict['len']['code'])
    f_dict['ratio']['ftc'] = ratio(f_dict['len']['firsttext'],
                                   f_dict['len']['firstcode'])
    f_dict['ratio']['ftext'] = ratio(f_dict['len']['firsttext'],
                                     f_dict['len']['text'])
    f_dict['ratio']['fcode'] = ratio(f_dict['len']['firstcode'],
                                     f_dict['len']['code'])
    f_dict['ratio']['qsent'] = ratio(f_dict['num']['question'],
                                     f_dict['num']['sent'])
    f_dict['ratio']['esent'] = ratio(f_dict['num']['exclam'],
                                     f_dict['num']['sent'])
    f_dict['ratio']['psent'] = ratio(f_dict['num']['period'],
                                     f_dict['num']['sent'])
    
    # mean features
    f_dict['mean']['code'] = np.mean([len(c) for c in code]) if code else 0
    f_dict['mean']['text'] = np.mean([len(t) for t in text]) if text else 0
    f_dict['mean']['sent'] = np.mean([len(s) for s in sents]) if sents else 0
    
    # user's post feature 
    f_dict['user'] = dict()
    post_time = dateparser.parse(datum['PostCreationDate'])
    user_create_time = dateparser.parse(datum['OwnerCreationDate'])
    f_dict['user']['age'] = (post_time - user_create_time).total_seconds()
    f_dict['user']['reputation'] = int(datum['ReputationAtPostCreation'])
    f_dict['user']['good_posts'] = int(datum['OwnerUndeletedAnswerCountAtPostTime'])

    return dict(f_dict)

def gen_datum_label(datum):
    try:
        post_status_id = status_map_label[datum['OpenStatus']]
    except KeyError:
        post_status_id = '0' # test set
    return post_status_id