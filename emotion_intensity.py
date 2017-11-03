import argparse
from collections import defaultdict
from itertools import takewhile

from numpy import average
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

from tweet_parser import *


def get_XY(tweets):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([t.message for t in tweets]).toarray()
    Y = [t.cl for t in tweets]
    return X, Y


def get_side(y, border):
    return int(y > border)


def get_classifier(X, Y, border):
    border_y = [get_side(y, border) for y in Y]
    clf = MultinomialNB()
    clf.fit(X, border_y)
    return clf


def get_answer(results):
    from_start = len(list(takewhile(lambda x: x == 1, results)))
    results.reverse()
    from_end = len(list(takewhile(lambda x: x == 0, results)))
    if (from_start + from_end) == len(results):
        return from_start
    else:
        return len(results) - from_end


def count_accuracy(classifiers, X, Y):
    total = len(Y)
    correct = 0
    for x, y in zip(X, Y):
        results = [clf.predict([x]) for clf in classifiers]
        final_result = get_answer(results)
        if final_result == y:
            correct += 1
    return correct / float(total)


def print_class_distribution():
    classes = defaultdict(int)
    for tweet in tweets:
        classes[tweet.cl] += 1
    print classes


def filter_index(X, index):
    return [X[i] for i in index]


def test_basic_classifier(train_X, train_Y, test_X, test_Y):
    comp_clf = MultinomialNB()
    comp_clf.fit(train_X, train_Y)
    print (comp_clf.score(test_X, test_Y))


def test_ordinal_classifier(train_X, train_Y, test_X, test_Y, accuracies):
    classifiers = [get_classifier(train_X, train_Y, b) for b in [0, 1, 2]]
    accuracies.append(count_accuracy(classifiers, test_X, test_Y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File name')
    parser.add_argument('-f', type=file, dest='data', help='data')
    args = parser.parse_args()

    tweets = get_tweets(args.data.read())
    print_class_distribution()

    X, Y = get_XY(tweets)
    kf = KFold(n_splits=10, shuffle=True)
    accuracies = []
    for train_index, test_index in kf.split(X):
        train_X = filter_index(X, train_index)
        train_Y = filter_index(Y, train_index)
        test_X = filter_index(X, test_index)
        test_Y = filter_index(Y, test_index)
        test_basic_classifier(train_X, train_Y, test_X, test_Y)
        test_ordinal_classifier(train_X, train_Y, test_X, test_Y, accuracies)
    print(average(accuracies))
