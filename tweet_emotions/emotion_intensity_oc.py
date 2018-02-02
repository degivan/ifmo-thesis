import argparse
from collections import defaultdict
from itertools import takewhile

from mord import *
from numpy import average
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from features.features import count_caps, count_symbol, count_intensity
from tweet_emotions.features.basic_vectorizers import get_XY_word_ngrams
from tweet_parser import *


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


def count_metrics(classifiers, X, Y):
    Y_pred = []
    for x, y in zip(X, Y):
        results = [clf.predict([x]) for clf in classifiers]
        final_result = get_answer(results)
        Y_pred.append(final_result)
    return accuracy_score(Y, Y_pred), f1_score(Y, Y_pred, average='macro')


def print_class_distribution():
    classes = defaultdict(int)
    for tweet in tweets:
        classes[tweet.res] += 1
    print classes
    print len(tweets)


def filter_index(X, index):
    return [X[i] for i in index]


def test_basic_classifier(train_X, train_Y, test_X, test_Y, metrics):
    comp_clf = MultinomialNB(alpha=1)
    comp_clf.fit(np.array(train_X), np.array(train_Y))
    predicted = comp_clf.predict(np.array(test_X))
    acc = accuracy_score(np.array(test_Y), predicted)
    f1 = f1_score(np.array(test_Y), predicted, average='macro')
    metrics.append((acc, f1))


def test_ordinal_classifier(train_X, train_Y, test_X, test_Y, metrics):
    classifiers = [get_classifier(train_X, train_Y, b) for b in [0, 1, 2]]
    metrics.append(count_metrics(classifiers, test_X, test_Y))


def test_mord_classifier(train_X, train_Y, test_X, test_Y, metrics):
    comp_clf = LogisticSE(max_iter=10 ** 6)
    comp_clf.fit(np.array(train_X), np.array(train_Y))
    predicted = comp_clf.predict(np.array(test_X))
    acc = accuracy_score(np.array(test_Y), predicted)
    f1 = f1_score(np.array(test_Y), predicted, average='macro')
    metrics.append((acc, f1))


def add_features(X, tweets, emotion):
    X_list = X.tolist()
    for x, tweet in zip(X_list, tweets):
        x.append(count_caps(tweet))
        x.append(count_symbol(tweet, '!'))
        x.append(count_intensity(tweet, emotion))
    return np.array(X_list)


def filter_tweets(tweets, noisy_words):
    result = []
    for tweet in tweets:
        msg = tweet.message
        for nw in noisy_words:
            msg = msg.replace(nw, '')
        result.append(Tweet(msg, tweet.res))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File name')
    parser.add_argument('-f', type=file, dest='data', help='data')
    parser.add_argument('-e', type=str, dest='emotion', help='emotion')
    args = parser.parse_args()
    tweets = get_tweets(args.data.read())
    problematic_words = {'but': 0, 'not': 0, 'although': 0}
    for tweet in tweets:
        for word in problematic_words.keys():
            if word in tweet.message:
                problematic_words[word] += 1
    print problematic_words
    tweets = filter_tweets(tweets, ['the'])
    X, Y = get_XY_word_ngrams(tweets)
    X = add_features(X, tweets, args.emotion)
    X = VarianceThreshold().fit_transform(X)
    Y = [int(y) for y in Y]
    kf = KFold(n_splits=10, shuffle=True)
    for test_classifier in [test_basic_classifier, test_ordinal_classifier, test_mord_classifier]:
        metrics = []
        for train_index, test_index in kf.split(X):
            train_X = filter_index(X, train_index)
            train_Y = filter_index(Y, train_index)
            test_X = filter_index(X, test_index)
            test_Y = filter_index(Y, test_index)
            test_classifier(train_X, train_Y, test_X, test_Y, metrics)
        accuracies = [x[0] for x in metrics]
        f1_scores = [x[1] for x in metrics]
        print ("Average accuracy:" + test_classifier.func_name + ": " + str(average(accuracies)))
        print ("Average F1-score:" + test_classifier.func_name + ": " + str(average(f1_scores)))
