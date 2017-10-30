class Tweet(object):
    def __init__(self, message, cl):
        self.message = message
        self.cl = cl


def get_tweet(str_tweet):
    num, message, common_class, cl = str_tweet.split('\t')
    return Tweet(message, int(cl[0]))


def get_tweets(str_tweets):
    return [get_tweet(line) for line in str_tweets.split('\n') if len(line) > 0]
