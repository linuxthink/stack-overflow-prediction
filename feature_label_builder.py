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
import string


def norm(string):
    return RE_NONANS.sub('', string).lower()

def norm_tag(string):
    return RE_NONALNUM.sub('', string).lower()

def ratio(x, y):
    if y != 0:
        return x / float(y)
    else:
        return 0

def dict_to_list(d):
    stack = []
    keys = sorted(d.keys())
    for k in keys:
        if type(d[k]) in (dict, defaultdict):
            stack += dict_to_list(d[k])
        else:
            stack.append(d[k])
    return stack

def dict_to_keys(d, prefix=''):
    stack = []
    keys = sorted(d.keys())
    for k in keys:
        prefix_k = k if len(prefix) == 0 else prefix + '-' + k
        if type(d[k]) in (dict, defaultdict):
            stack += dict_to_keys(d[k], prefix_k)
        else:
            stack.append(prefix_k)
    return stack

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


class FeatureLabelBuilder(object):
    """
    Usage:
        flb = FeatureLabelBuilder(datum)
        # usually
        flb.feature
        flb.label
        # can return feature as dict
        flb.feature_dict
    """

    def __init__(self, datum):
        self.datum = datum
        self.feature_nested_dict = FeatureLabelBuilder.__get_feature_nested_dict(self.datum)
        self.label = FeatureLabelBuilder.__get_label(self.datum)
        self.feature = dict_to_list(self.feature_nested_dict)
        self.feature_keys_cache = None # to be computed on demand
        self.feature_dict_cache = None # to be computed on demand

    @property
    def feature_keys(self):
        if not self.feature_keys_cache:
            self.feature_keys_cache = dict_to_keys(self.feature_nested_dict)
        return self.feature_keys_cache

    @property
    def feature_dict(self):
        if not self.feature_dict_cache:
            d = dict()
            keys = self.feature_keys
            vals = self.feature
            for k, v in zip(keys, vals):
                d[k] = v
            self.feature_dict_cache = d
        return self.feature_dict_cache

    @staticmethod
    def __get_feature_nested_dict(datum):
        """
        return feature in a dict format
        """

        # feature container
        f_dict = defaultdict(dict)

        # get text features
        body = datum['BodyMarkdown']
        lines = body.splitlines()
        code = [] # code
        text = [] # text
        sentences = [] # sentence
        title = datum['Title'] # title
        tags = [norm_tag(datum["Tag%d" % i])
                for i in range(1, 6) if datum["Tag%d" % i]]

        # divide post into code and text blocks
        for is_code, group in groupby(lines, lambda l: l.startswith('    ')):
            (code if is_code else text).append('\n'.join(group))

        # build text f_dict features
        f_dict['num']['sentence'] = 0
        f_dict['num']['question'] = 0
        f_dict['num']['exclam'] = 0
        f_dict['num']['period'] = 0
        f_dict['num']['init_cap'] = 0
        f_dict['num']['i_start'] = 0
        f_dict['num']['url'] = 0
        f_dict['num']['digit'] = 0
        f_dict['num']['non_word'] = 0

        for t in text:
            # try:
            #     nltk.sent_tokenize(t)
            # except:
            #     import ipdb; ipdb.set_trace()
            #     print(text)
            #     raise

            for sent in nltk.sent_tokenize(filter(lambda x: x in string.printable, t)):
                f_dict['num']['sentence'] += 1
                ss = sent.strip()
                if ss:
                    if ss.endswith('?'):
                        f_dict['num']['question'] += 1
                    if ss.endswith('!'):
                        f_dict['num']['exclam'] += 1
                    if ss.endswith('.'):
                        f_dict['num']['period'] += 1
                    if ss.startswith('I '):
                        f_dict['num']['i_start'] += 1
                    if ss[0].isupper():
                        f_dict['num']['init_cap'] += 1

                words = nltk.word_tokenize(norm(sent))
                sentences.append(ss)

            f_dict['num']['digit'] += len(RE_DIGIT.findall(t))
            f_dict['num']['url'] += len(RE_URL.findall(t))
            f_dict['num']['non_word'] += len(RE_NONWORD.findall(t))

        f_dict['num']['final_thanks'] = 1 if text and 'thank' in text[-1].lower() else 0
        f_dict['num']['code_block'] = len(code)
        f_dict['num']['text_block'] = len(text)
        f_dict['num']['lines'] = len(lines)
        f_dict['num']['tags'] = len(tags)

        # len features
        f_dict['len']['title'] = len(title)
        f_dict['len']['text'] = sum(len(t) for t in text)
        f_dict['len']['code'] = sum(len(c) for c in code)
        f_dict['len']['first_text'] = len(text[0]) if text else 0
        f_dict['len']['first_code'] = len(code[0]) if code else 0
        f_dict['len']['last_text'] = len(text[-1]) if text else 0
        f_dict['len']['last_code'] = len(code[-1]) if code else 0

        # ratio features
        f_dict['ratio']['text_code'] = ratio(f_dict['len']['text'],
                                      f_dict['len']['code'])
        f_dict['ratio']['first_text_first_code'] = ratio(f_dict['len']['first_text'],
                                                         f_dict['len']['first_code'])
        f_dict['ratio']['first_text_text'] = ratio(f_dict['len']['first_text'],
                                                   f_dict['len']['text'])
        f_dict['ratio']['first_code_code'] = ratio(f_dict['len']['first_code'],
                                                   f_dict['len']['code'])
        f_dict['ratio']['question_sentence'] = ratio(f_dict['num']['question'],
                                                     f_dict['num']['sentence'])
        f_dict['ratio']['exclam_sentence'] = ratio(f_dict['num']['exclam'],
                                                   f_dict['num']['sentence'])
        f_dict['ratio']['period_sentence'] = ratio(f_dict['num']['period'],
                                                   f_dict['num']['sentence'])

        # mean features
        f_dict['mean']['code'] = np.mean([len(c) for c in code]) if code else 0
        f_dict['mean']['text'] = np.mean([len(t) for t in text]) if text else 0
        f_dict['mean']['sentence'] = np.mean(
            [len(s) for s in sentences]) if sentences else 0

        # user's post feature
        f_dict['user'] = dict()
        post_time = dateparser.parse(datum['PostCreationDate'])
        user_create_time = dateparser.parse(datum['OwnerCreationDate'])
        f_dict['user']['age'] = (post_time - user_create_time).total_seconds()
        f_dict['user']['reputation'] = int(datum['ReputationAtPostCreation'])
        f_dict['user']['good_posts'] = int(
            datum['OwnerUndeletedAnswerCountAtPostTime'])

        # time
        f_dict['time']['year'] = post_time.year
        f_dict['time']['month'] = post_time.month
        f_dict['time']['day'] = post_time.day
        f_dict['time']['weekday'] = post_time.weekday()

        return dict(f_dict)

    @staticmethod
    def __get_label(datum):
        """
        return label as int 0,1,2,3,4,5; 0 if test set
        """

        try:
            label = status_map_label[datum['OpenStatus']]
        except KeyError:
            label = 0  # test set
        return label