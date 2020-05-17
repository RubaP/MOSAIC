import tweepy
import numpy as np
import pandas as pd
from tweepy import OAuthHandler
import re
import json

def cleanTweets(tweets):
    results = np.empty(shape=(0,15), dtype=object)
    count = 0
    for tweet in tweets:
        tweet = tweet._json

        if tweet['lang'] != 'en':
            continue

        id = tweet['id_str']
        time = tweet['created_at']
        userR = tweet['user']['id']
        reply = False
        quotedText = ""
        retweeted = False
        quoted = False

        if 'retweeted_status' in tweet:
            tweet = tweet['retweeted_status']
            retweeted = True
        if 'quoted_status' in tweet:
            quotedText = tweet['text']
            tweet = tweet['quoted_status']
            quoted = True
        if tweet['in_reply_to_status_id'] != None:
            reply = True

        created_time = tweet['created_at']
        text = re.sub('\s+', ' ', tweet['text'])
        original_tweetId = tweet['id']
        retweet_count = tweet['retweet_count']
        favorite_count = tweet['favorite_count']
        user = tweet['user']['id']
        location = tweet['user']['location']
        followers = tweet['user']['followers_count']

        results = np.append(results, np.array((id, text, retweeted, retweet_count, favorite_count, time, followers, original_tweetId,
                     reply, quoted, quotedText, created_time, user, location, userR), ndmin=2, dtype=object), axis=0)
        count += 1
    return results

def saveTweets(tweets, count):
    data = []
    for tweet in tweets:
        if tweet.lang != 'en':
            continue
        data.append(tweet._json)
    with open("PATH_TO_SAVE_TWEET_JSONS" + str(count) + '.json', 'w') as fp:
        json.dump(data, fp)

with open("PATH_TO_TEXT_FILE_WITH_TWEETIDS") as f:
    content = f.readlines()

content = [x.strip() for x in content]
noOfTweets = len(content)
print("File reading completed. Recived ",noOfTweets, "Tweets")

consumer_key = 'YOUR_KEY'
consumer_secret = 'YOUR_SECRET'
access_token = 'YOUR_TOKEN'
access_secret = 'YOUR_SECRET'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, retry_count=300, retry_delay=30,
                 retry_errors=set([401, 404, 500, 503]), timeout=60, wait_on_rate_limit=True)
print("Authentication successful")

retrievedCount = 0
availableTweets = 0
nonEnglishCount = 0
results = np.empty(shape=(0,15), dtype=object)

while retrievedCount < noOfTweets:
  if(retrievedCount + 100 > noOfTweets):
      lastIndex = noOfTweets
  else:
      lastIndex = retrievedCount + 100
  try:
    tweets = api.statuses_lookup(content[retrievedCount:lastIndex])
    saveTweets(tweets, retrievedCount)
    availableTweets += len(tweets)
  except Exception as e:
      print(e)
      continue
  data = cleanTweets(tweets)
  results = np.append(results, data, axis=0)
  retrievedCount += 100
  print("Processed upto : ", retrievedCount)

print("Avaiable tweets: ", availableTweets)
df = pd.DataFrame(results)
df.to_csv("FILE_NAME.csv", header=["id", "text", "retweeted", "retweet_count", "favorite_count", "time",
                                   "followers", "original_tweetId", "reply", "quoted", "quotedText",
                                         "created_time", "user", "location", "userR"], index_label="No", encoding="utf")


