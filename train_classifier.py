from __future__ import print_function
from pprint import pprint
import os
import csv
import cPickle as pickle

import numpy as np
import scipy as sp
import sklearn
from sklearn.ensemble import GradientBoostingClassifier as GBC

from feature_label_builder import FeatureLabelBuilder, get_feature_keys
from feature_label_builder import data_path, full_path

(train_xs, train_ys) = pickle.load(open(full_path("train_xs_ys_np.p"), "rb"))

classifier = GBC(n_estimators=2000,
                 learning_rate=0.1,
                 subsample=0.001,
                 max_features='auto',
                 min_samples_leaf=9,
                 verbose=1)
classifier.fit(train_xs, train_ys)

pickle.dump(classifier, open(full_path("classifier_2000_0.1_0.001_9.p"), "wb"))

# keys = get_feature_keys()
# for key, val in zip(keys, classifier.feature_importances_):
#     print(key, val)

