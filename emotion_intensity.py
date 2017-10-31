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


def get_side(y, border):
    return int(y > border)


def get_classifier(X, Y, border):
    border_y = [get_side(y, border) for y in Y]
    clf = GaussianNB()
    clf.fit(X, border_y)
    return clf


def count_accuracy(classifiers, X, Y):
    total = len(Y)
    correct = 0
    for x, y in zip(X, Y):
        results = [clf.predict([x]) for clf in classifiers]
        final_result = 0
        while final_result < 3:
            if results[final_result] == 1:
                final_result += 1
            else:
                break
        if final_result == y:
            correct += 1
    return correct / float(total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File name')
    parser.add_argument('-f', type=file, dest='data', help='data')
    args = parser.parse_args()
    tweets = get_tweets(args.data.read())
    X, Y = get_XY()
    kf = KFold(n_splits=10, shuffle=True)
    accuracies = []
    for train_index, test_index in kf.split(X):
        train_X = [X[i] for i in train_index]
        train_Y = [Y[i] for i in train_index]
        test_X = [X[i] for i in test_index]
        test_Y = [Y[i] for i in test_index]
        classifiers = [get_classifier(train_X, train_Y, b) for b in [0, 1, 2]]
        accuracies.append(count_accuracy(classifiers, test_X, test_Y))
    print(average(accuracies))