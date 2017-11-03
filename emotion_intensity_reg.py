import argparse

from tweet_parser import get_tweets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File name')
    parser.add_argument('-f', type=file, dest='data', help='data')
    args = parser.parse_args()

    tweets = get_tweets(args.data.read(), res_acc=5)
    for tweet in tweets:
        print tweet
