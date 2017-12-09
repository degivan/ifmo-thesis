class Tweet(object):
    def __init__(self, message, res):
        self.message = message
        self.res = res

    def __str__(self):
        return str(self.message) + " " + str(self.res)


def get_tweet(str_tweet, res_acc=1):
    num, message, common_class, res = str_tweet.split('\t')
    return Tweet(message, float(res[0:res_acc]))


def get_tweets(str_tweets, res_acc=1):
    return [get_tweet(line, res_acc) for line in str_tweets.split('\n') if len(line) > 0]
