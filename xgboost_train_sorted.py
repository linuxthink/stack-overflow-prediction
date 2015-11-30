from __future__ import print_function
from pprint import pprint
import os
import csv
import cPickle as pickle

import numpy as np
import scipy as sp

from feature_label_builder import FeatureLabelBuilder, get_feature_keys
from feature_label_builder import data_path, full_path
from util_eval import multiclass_log_loss, multiclass_accuracy

import xgboost as xgb
import itertools

# load data
(all_xs, all_ys) = pickle.load(open(full_path("train_37_xs_ys_np_sorted.p"), "rb"))
all_ys = all_ys - 1.0

# set up split portion of train and test data
test_percentage = 0.01
all_num = len(all_ys)
train_num = int(round((1. - test_percentage) * all_num))
test_num = all_num - train_num

# prepare train and test dataset
train_xs = all_xs[:train_num]
train_ys = all_ys[:train_num]
test_xs = all_xs[train_num:]
test_ys = all_ys[train_num:]

# setup param grid
params_grid = dict()
params_grid['bst:max_depth'] = [3, 4, 5]
params_grid['bst:eta'] = [0.01, 0.02, 0.05, 0.1, 0.2]
params_grid['subsample'] = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

# set up boosting params
params = {'bst:max_depth': None,
          'bst:eta': None,
          'silent': 0,
          'objective': 'multi:softprob',
          'num_class': 5,
          'nthread': 32, # change this!
          'eval_metric': 'mlogloss',
          'subsample': None}

ops = {'num_boost_round': 2000,
       'early_stopping_rounds': 10}

print("start training")

for d, e, s in itertools.product(params_grid['bst:max_depth'],
                                 params_grid['bst:eta'],
                                 params_grid['subsample']):
    params['bst:max_depth'] = d
    params['bst:eta'] = e
    params['subsample'] = s

    print(params)

    # convert to xgb matrix, reload each time
    dtrain = xgb.DMatrix(train_xs, label=train_ys)
    dtest = xgb.DMatrix(test_xs, label=test_ys)

    bst = xgb.train(params.items(),
                    dtrain,
                    num_boost_round=ops['num_boost_round'],
                    evals=[(dtest, 'eval'), (dtrain, 'train')],
                    early_stopping_rounds=ops['early_stopping_rounds'])

    # load test (valid) set
    dtest = xgb.DMatrix(test_xs)

    # predict
    ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
    test_ys_preds = np.array(ypred)

    # metrics
    log_loss = multiclass_log_loss(test_ys, test_ys_preds)
    print(log_loss)
    print(multiclass_accuracy(test_ys, test_ys_preds))

    # dump model
    bst.dump_model('xgboost_sorted_round_%s_%s_%s_%s_%s_%s.txt'
                   % (ops['num_boost_round'],
                      ops['early_stopping_rounds'],
                      params['bst:max_depth'],
                      params['bst:eta'],
                      params['subsample'],
                      log_loss))
    bst.save_model('xgboost_sorted_round_%s_%s_%s_%s_%s_%s.model'
                   % (ops['num_boost_round'],
                      ops['early_stopping_rounds'],
                      params['bst:max_depth'],
                      params['bst:eta'],
                      params['subsample'],
                      log_loss))
# import ipdb; ipdb.set_trace()

