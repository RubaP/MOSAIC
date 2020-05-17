import json
import pandas as pd
import time
import math

def readConfig():
    with open('../../config.json') as config_file:
        config = json.load(config_file)
    return config

def readAndProcessFile(config):
    df = readFile(config)
    df = addScores(df)
    df = addTimestamp(df, config)
    df = cleanColumns(df)
    return df

def readFile(config):
    print("=========STARTED FILE READING========")
    path = config['path'] + config['event_name'] + ".csv"
    print("path: ", path)
    df = pd.read_csv(path, index_col=0, encoding="utf")
    df = df[pd.notnull(df['id'])]
    df = df[pd.notnull(df['original_tweetId'])]
    df['original_tweetId'] = df['original_tweetId'].apply(lambda x: str(int(x)))
    totalTweets = df.shape[0]
    print("Total Tweets", totalTweets)
    print("========FILE READING COMPLETED=======")
    return df

def addTimestamp(df, config):
    df['timestamp'] = df['time'].apply(lambda x: int(time.mktime(time.strptime(x, config['timestamp_format']))))
    return df

def addScores(df):
    df['followers_log'] = df['followers'].apply(lambda x: 0 if x <= 0 else math.log(x))
    df['retweet_count_log'] = df['retweet_count'].apply(lambda x: 0 if x <= 0 else math.log(x))
    print("followers_log max score: ", df['followers_log'].max())
    print("retweet_count_log max score: ", df['retweet_count_log'].max())
    return df

def cleanColumns(df):
    df = df.drop(['retweeted', 'retweet_count', 'favorite_count', 'time', 'followers'], axis=1)
    print("Remaining columns after removal", df.columns)
    return df

def removeRetweet(df):
    print("==========REMOVING RETWEETS==========")
    df = df.drop_duplicates(subset='original_tweetId', keep="first")
    df = df.drop(['id'], axis=1)
    print("Removed duplicate ")
    print("Size of the dataset after dropping retweets: ", df.shape[0])
    print("==========RETWEETS REMOVED===========")
    return df