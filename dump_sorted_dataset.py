from __future__ import print_function
from pprint import pprint
import os
import csv
import cPickle as pickle
from dateutil import parser as dateparser
import datetime, time

import numpy as np
import scipy as sp

from feature_label_builder import FeatureLabelBuilder, get_feature_keys
from feature_label_builder import data_path, full_path

# reader = csv.DictReader(open(os.path.join(data_path, 'train_October_9_2012.csv')))
# post_times = [int(time.mktime(dateparser.parse(datum['PostCreationDate']).timetuple()))
#               for datum in reader]
# post_times = np.array(post_times)
# pickle.dump(post_times, open(full_path("post_times.p"), "wb"),
#             protocol=pickle.HIGHEST_PROTOCOL)
post_times = np.array(pickle.load(open(full_path("post_times.p"), "rb")))
sort_arg = np.argsort(post_times)


def is_sorted(l):
    return all(l[i] <= l[i+1] for i in xrange(len(l)-1))

print(is_sorted(post_times))
print(is_sorted(post_times[sort_arg]))

(train_xs, train_ys) = pickle.load(open(full_path("train_xs_ys_np.p"), "rb"))
train_xs = train_xs[sort_arg]
train_ys = train_ys[sort_arg]
pickle.dump((train_xs, train_ys),
            open(full_path("train_xs_ys_np_sorted.p"), "wb"),
            protocol=pickle.HIGHEST_PROTOCOL)

