from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt


from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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


useTFIDF = True
showSampleVector = False
showMostInformativeFeatures = True
howManyInformativeFeatures = 20
nGRAM1 = 8
nGRAM2 = 10
weight = 4

ask = input("Do you want to specify parameters or use default values? Input 'T' or 'F'.   ")
if ask == "T":
    useTFIDFStr = input("Do you want to use tfidfVectorizer or CountVectorizer? Type T for tfidfVectorizer and F for CountVectorizer   ")
    if useTFIDFStr == "T":
        useTFIDF = True
    else:
        useTFIDF = False

    showSampleVectorStr = input("Do you want to print an example vectorized corpus? (T/F)   ")
    if showSampleVectorStr == "T":
        showSampleVector = True
    else:
        showSampleVector = False

    showMostInformativeFeaturesStr = input("Do you want to print the most informative feature in some of the classifiers? (T/F)   ")
    if showMostInformativeFeaturesStr == "T":
        showMostInformativeFeatures = True
        howManyInformativeFeatures = int(input("How many of these informative features do you want to print for each binary case? Input a number   "))
    else:
        showMostInformativeFeatures = False

    nGRAM1 = int(input("N-Gram lower bound (Read README.md for more information)? Input a number   "))
    nGRAM2 = int(input("N-Gram Upper bound? Input a number   "))
    weight = int(input("What weight do you want to use to separate train & testing? Input a number   "))


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



print("Extracting features from the training data using a sparse vectorizer...")
t0 = time()

if useTFIDF:
    vectorizer = TfidfVectorizer(ngram_range=(nGRAM1, nGRAM2), min_df=1, use_idf=True, smooth_idf=True) ##############
else:
    vectorizer = CountVectorizer(ngram_range=(nGRAM1, nGRAM2))

analyze = vectorizer.build_analyzer()


X_train = vectorizer.fit_transform(train_corpus)

if showSampleVector:
    print(vectorizer.get_feature_names())
    print(X_train.toarray()[0])


duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, train_corpus_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer...")
t0 = time()
X_test = vectorizer.transform(test_corpus)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, test_corpus_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()


# show which are the definitive features
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    coefs_with_fns_mal = coefs_with_fns[:-(n + 1):-1]
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))[:n]

    print()
    print("Most Informative Benign Features:")
    for (coef_1, fn_1) in coefs_with_fns:
        print(coef_1, fn_1)
    print()
    print("Most Informative Malicious Features:")
    for (coef_2, fn_2) in coefs_with_fns_mal:
        print(coef_2, fn_2)
    print()



test_corpus_target = np.array(test_corpus_target)



def benchmark(clf, showTopFeatures=False):
    print('_'*60)
    print("Training: ")
    print(clf)
    probas_ = clf.fit(X_train, train_corpus_target).predict_proba(X_test)

    # print("shape test_corpus_target: ", np.shape(test_corpus_target))
    # print("shape probas: ", np.shape(probas_[:, 0]))

    print("test_corpus_target:", np.array(test_corpus_target))
    print("probas: ", probas_[:, 1])

    fpr, tpr, _ = roc_curve(np.array(test_corpus_target), probas_[:, 1], pos_label=1, drop_intermediate=False)
    print("thresholds: ", _)
    roc_auc = auc(fpr, tpr)

    # Compute micro-average ROC curve and ROC area
    print("fpr", fpr)
    print("tpr", tpr)
    print("roc", roc_auc)
    # printing ROC
    plt.figure()
    lwr = 2
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lwr, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=0)
benchmark(classifier)

