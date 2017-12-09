def count_caps(tweet):
    caps = 0
    for word in tweet.message.split():
        caps += (len(word) > 2) & (word.isupper())
    return caps


def count_symbol(tweet, symbol):
    return tweet.message.count(symbol)


def starts_with_vowel(tweet):
    return tweet.message[0] in 'AaIiEeUuOo'
