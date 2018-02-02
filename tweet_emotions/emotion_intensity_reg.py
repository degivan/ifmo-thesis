import argparse

from tweet_parser import get_tweets
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import ntpath


def draw_distribution(name):
    emotion_name = ntpath.basename(name).split('_')[1]
    results = [t.res for t in tweets]
    (mu, sigma) = norm.fit(results)
    plt.xlabel('Intensity')
    n, bins, patches = plt.hist(results, 70, normed=1, facecolor='green', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=2)
    plt.grid(True)
    plt.savefig('distribution_%s.png' % emotion_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File name')
    parser.add_argument('-f', type=file, dest='data', help='data')
    args = parser.parse_args()

    tweets = get_tweets(args.data.read(), res_acc=5)
    draw_distribution(args.data.name)
