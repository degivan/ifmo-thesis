import argparse

from numpy import average
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

from tweet_parser import *


def get_XY():
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([t.message for t in tweets]).toarray()
    Y = [t.cl for t in tweets]
    return X, Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File name')
    parser.add_argument('-f', type=file, dest='data', help='data')
    args = parser.parse_args()
    tweets = get_tweets(args.data.read())
    X, Y = get_XY()
    kf = KFold(n_splits=10, shuffle=True)
    accuracies = []
    for train_index, test_index in kf.split(X):
        clf = GaussianNB()
        train_X = [X[i] for i in train_index]
        train_Y = [Y[i] for i in train_index]
        test_X = [X[i] for i in test_index]
        test_Y = [Y[i] for i in test_index]
        clf.fit(train_X, train_Y)
        accuracies.append(clf.score(test_X, test_Y))
    print(average(accuracies))
