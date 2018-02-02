from sklearn.feature_extraction.text import CountVectorizer


def get_XY_char_ngrams(tweets):
    vectorizer = CountVectorizer(analyzer='char_wb', max_features=1000, ngram_range=(3, 8))
    X = vectorizer.fit_transform([t.message for t in tweets]).toarray()
    Y = [t.res for t in tweets]
    return X, Y

def get_XY_word_ngrams(tweets):
    vectorizer = CountVectorizer(max_features=500)
    X = vectorizer.fit_transform([t.message for t in tweets]).toarray()
    Y = [t.res for t in tweets]
    return X, Y
