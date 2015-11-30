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

# load data
(all_xs, all_ys) = pickle.load(open(full_path("train_xs_ys_np.p"), "rb"))
all_ys = all_ys - 1.0

# set up split portion of train and test data
test_percentage = 0.1
all_num = len(all_ys)
train_num = int(round((1. - test_percentage) * all_num))
test_num = all_num - train_num

# prepare random shuffle index
random_idx = np.array(range(all_num))
np.random.seed(0)
np.random.shuffle(random_idx)

# prepare train and test dataset
train_xs = all_xs[random_idx][:train_num]
train_ys = all_ys[random_idx][:train_num]
test_xs = all_xs[random_idx][train_num:]
test_ys = all_ys[random_idx][train_num:]

# convert to xgb matrix
dtrain = xgb.DMatrix(train_xs, label=train_ys)
dtest = xgb.DMatrix(test_xs, label=test_ys)

# set up boosting params
params = {'bst:max_depth': 5,
          'bst:eta': 0.1,
          'silent': 0,
          'objective': 'multi:softprob',
          'num_class': 5,
          'nthread': 16, # change this!
          'eval_metric': 'mlogloss',
          'subsample': 0.01}

ops = {'num_boost_round': 2000,
       'early_stopping_rounds': 10}

print("start training")
bst = xgb.train(params.items(),
                dtrain,
                num_boost_round=ops['num_boost_round'],
                evals=[(dtest, 'eval'), (dtrain, 'train')],
                early_stopping_rounds=ops['early_stopping_rounds'])

# dump model
bst.dump_model('xgboost_round_%s_%s.txt' % (ops['num_boost_round'],
                                            ops['early_stopping_rounds']))
bst.save_model('xgboost_round_%s_%s.model' % (ops['num_boost_round'],
                                              ops['early_stopping_rounds']))

# load test (valid) set
dtest = xgb.DMatrix(test_xs)

# predict
ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
test_ys_preds = np.array(ypred)

# metrics
print(multiclass_log_loss(test_ys, test_ys_preds))
print(multiclass_accuracy(test_ys, test_ys_preds))

# import ipdb; ipdb.set_trace()

