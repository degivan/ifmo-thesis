import argparse
from collections import defaultdict
from itertools import takewhile

from mord import *
from numpy import average
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB

from tweet_emotions.features import count_caps, count_symbol, starts_with_vowel
from tweet_parser import *


def get_XY(tweets):
    vectorizer = CountVectorizer(max_features=900, ngram_range=(1, 3))
    X = vectorizer.fit_transform([t.message for t in tweets]).toarray()
    Y = [t.res for t in tweets]
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
        classes[tweet.res] += 1
    print classes
    print len(tweets)


def filter_index(X, index):
    return [X[i] for i in index]


def test_basic_classifier(train_X, train_Y, test_X, test_Y, accuracies):
    comp_clf = MultinomialNB()
    comp_clf.fit(np.array(train_X), np.array(train_Y))
    predicted = comp_clf.predict(np.array(test_X))
    acc = metrics.accuracy_score(np.array(test_Y), predicted)
    accuracies.append(acc)


def test_ordinal_classifier(train_X, train_Y, test_X, test_Y, accuracies):
    classifiers = [get_classifier(train_X, train_Y, b) for b in [0, 1, 2]]
    accuracies.append(count_accuracy(classifiers, test_X, test_Y))


def test_mord_classifier(train_X, train_Y, test_X, test_Y, accuracies):
    comp_clf = LogisticSE(alpha=0, max_iter=1000)
    comp_clf.fit(np.array(train_X), np.array(train_Y))
    predicted = comp_clf.predict(np.array(test_X))
    acc = metrics.accuracy_score(np.array(test_Y), predicted)
    accuracies.append(acc)


def add_features(X, tweets):
    X_list = X.tolist()
    for x, tweet in zip(X_list, tweets):
        x.append(count_caps(tweet))
        x.append(count_symbol(tweet, '!'))
        # x.append(starts_with_vowel(tweet))
    return np.array(X_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File name')
    parser.add_argument('-f', type=file, dest='data', help='data')
    args = parser.parse_args()

    tweets = get_tweets(args.data.read())
    print_class_distribution()

    X, Y = get_XY(tweets)
    X = add_features(X, tweets)
    Y = [int(y) for y in Y]
    kf = KFold(n_splits=10, shuffle=True)
    basic_accuracies = []
    ord_accuracies = []
    # mord_accuracies = []
    for train_index, test_index in kf.split(X):
        train_X = filter_index(X, train_index)
        train_Y = filter_index(Y, train_index)
        test_X = filter_index(X, test_index)
        test_Y = filter_index(Y, test_index)
        test_basic_classifier(train_X, train_Y, test_X, test_Y, basic_accuracies)
        test_ordinal_classifier(train_X, train_Y, test_X, test_Y, ord_accuracies)
        # test_mord_classifier(train_X, train_Y, test_X, test_Y, mord_accuracies)
    print("Average basic: " + str(average(basic_accuracies)))
    print("Average ordinal: " + str(average(ord_accuracies)))
    # print("Average mord: " + str(average(mord_accuracies)))
