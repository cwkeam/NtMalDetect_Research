from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.utils import shuffle
from pprint import pprint
from time import time
import logging

useTFIDF = True
showSampleVector = False
showMostInformativeFeatures = True
howManyInformativeFeatures = 10
nGRAM1 = 10
nGRAM2 = 10
weight = 10

main_corpus = []
main_corpus_target = []

my_categories = ['benign', 'malware']

# feeding corpus the testing data

print("Loading system call database for categories:")
print(my_categories if my_categories else "all")


import glob
import os

malCOUNT = 0
benCOUNT = 0
for filename in glob.glob(os.path.join('./sysMAL', '*.txt')):
    fMAL = open(filename, "r")
    aggregate = ""
    for line in fMAL:
        linea = line[:(len(line)-1)]
        aggregate += " " + linea
    main_corpus.append(aggregate)
    main_corpus_target.append(1)
    malCOUNT += 1

for filename in glob.glob(os.path.join('./sysBEN', '*.txt')):
    fBEN = open(filename, "r")
    aggregate = ""
    for line in fBEN:
        linea = line[:(len(line) - 1)]
        aggregate += " " + linea
    main_corpus.append(aggregate)
    main_corpus_target.append(0)
    benCOUNT += 1

# shuffling the dataset
main_corpus_target, main_corpus = shuffle(main_corpus_target, main_corpus, random_state=0)


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


# weight as determined in the top of the code
train_corpus = main_corpus[:(weight*len(main_corpus)//(weight+1))]
train_corpus_target = main_corpus_target[:(weight*len(main_corpus)//(weight+1))]
test_corpus = main_corpus[(len(main_corpus)-(len(main_corpus)//(weight+1))):]
test_corpus_target = main_corpus_target[(len(main_corpus)-len(main_corpus)//(weight+1)):]

# size of datasets
train_corpus_size_mb = size_mb(train_corpus)
test_corpus_size_mb = size_mb(test_corpus)


print("%d documents - %0.3fMB (training set)" % (
    len(train_corpus_target), train_corpus_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(test_corpus_target), test_corpus_size_mb))
print("%d categories" % len(my_categories))
print()
print("Benign Traces: "+str(benCOUNT)+" traces")
print("Malicious Traces: "+str(malCOUNT)+" traces")
print()


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC()),
])

parameters = {
    'vect__max_df': (0.75, 1.0),
    'vect__ngram_range': ((7, 7), (8, 8), (9, 9)),
    'tfidf__norm': ('l1', 'l2'),
    'clf__penalty': ('l2', 'l1'),
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(train_corpus, train_corpus_target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = test_corpus_target, grid_search.predict(test_corpus)
    print(classification_report(y_true, y_pred))
    print()


