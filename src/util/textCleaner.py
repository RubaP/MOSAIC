import src.components.preprocessor as p
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import collections
from nltk.corpus import stopwords
import math
import twikenizer as twk
import datetime
twk = twk.Twikenizer()

stop_words = stopwords.words('english')

lemmatizer = WordNetLemmatizer()

p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG)
tokenizer = RegexpTokenizer(r'\w+')

"""
Clean the complete data for learning
"""
def clean(data):
    print("------STARTED CLEANING TEXT DATA-------")
    data = data.apply(lambda x: cleadData(x), axis=1)
    print("------TEXT DATA CLEANING FINISHED-------", datetime.datetime.now())
    return data

"""
Clean a tweet and return a list of words. This includes hashtags and emojis and excludes user mention and url
"""
def cleanTweet(tweet):
    text = tweet.lower()
    emojis, hashtags = extractEmojisAndHastags(text)
    cleaned = tokenize(p.clean(text))
    all = cleaned.copy()
    all.extend(hashtags)
    all.extend(emojis)
    return all

"""
Add a column called cleanDataPresentation without quoted text and emojis and hastag available in quoted text
"""
def cleanDataForPresentation(data):
    text = data['text']
    text = text.lower()
    emojis, hashtags = extractEmojisAndHastags(text)
    data['emojisPresentation'] = emojis
    data['hashtagsPresentation'] = hashtags
    cleaned = tokenize(p.clean(text))
    data['cleanedTextPresentation'] = cleaned
    all = cleaned.copy()
    all.extend(hashtags)
    all.extend(emojis)
    data['cleanDataPresentation'] = all
    return data

"""
Add cleanedText without hashtag and emoji and cleanedCompleteText with hashtag and emoji.
Both columns include quotedText and only for information learning purpose 
"""
def cleadData(data):
    if type(data['quotedText']) == float and math.isnan(data['quotedText']):
        text = data['text']
    else:
        text = data['text'] + " " + data['quotedText']
    text = text.lower()
    emojis, hashtags = extractEmojisAndHastags(text)
    data['emojis'] = emojis
    data['hashtags'] = hashtags
    cleaned = tokenize(p.clean(text))
    data['cleanedText'] = cleaned
    all = cleaned.copy()
    all.extend(hashtags)
    all.extend(emojis)
    data['cleanedCompleteText'] = all
    return data

def filterSmallDocs(df):
    df['doc_len'] = df['cleanedCompleteText'].apply(lambda x: len(x))
    data = df[df.doc_len >= 3]

    print("Number of documents after filteration: ", data.shape[0])
    return data

def extractEmojisAndHastags(tweet):
    parsed_tweet = p.parse(tweet)
    emojis = []
    hashtags = []
    if parsed_tweet.emojis is not None:
        for emoji in parsed_tweet.emojis:
            emojis.append(emoji.match)
    if parsed_tweet.hashtags is not None:
        for hashtag in parsed_tweet.hashtags:
            hashtags.append(lemmatizer.lemmatize(hashtag.match[1:]))
    return emojis, hashtags

def tokenize(text):
    tokens = tokenizer.tokenize(text)
    result = []
    for token in tokens:
        if "..." in token or token == 'gt' or token == 'amp':
            continue
        elif token.isnumeric():
            result.append(token)
        elif len(token) == 1:
            continue
        elif token not in stop_words:
            result.append(lemmatizer.lemmatize(token))
    return result

"""
Get hashtag list for an interval
"""
def getHashtagList(hashtags):
    list = []
    hashtags.apply(lambda x: list.extend(x))
    counter = collections.Counter(list)
    hashtags = []

    for key in counter:  # list is important here
        cnts = counter[key]
        if key in stop_words or len(key) == 1:
            continue
        elif cnts >= 10: #a hashtag should appear atleast 10 times as they are more noisy compare to normal words
            hashtags.append(key)

    print("Number of hastags: ", len(hashtags))
    return hashtags, counter