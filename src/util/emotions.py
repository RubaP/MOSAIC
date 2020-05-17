import json
import pandas as pd
import collections

"""
Get the complete emoji vocabulary. This includes emoji and corresponding probability distribution
"""
def get_emojiSeedWords(config):
    df = pd.read_csv(config['emoji_vocabulary_path'], index_col=0, encoding="utf")
    df['probability'] = df.apply(lambda x: ['%.4f' % elem for elem in list(x[1:8].astype(float) / x[1:8].astype(float).sum())], axis=1)
    df = df[['emoji', 'probability']]
    keys = [i[0] for i in df[['emoji']].values]
    values = [i[0] for i in df[['probability']].values]
    emoji = dict(zip(keys, values))
    print("----------Emoji Vocabulary is read-------")
    return emoji

"""
Get emoji list given the emoji column of the dataset
"""
def getEmojiList(emojis):
    listE = []
    emojis.apply(lambda x: listE.extend(x))
    counter = collections.Counter(listE)
    print(counter)
    print("Number of emojis: ", len(counter.keys()))

    uniqueEmojis = []
    for key in counter:  # list is important here
        cnts = counter[key]
        if cnts >= 1:
            if key != '🏿' and key != '🏾' and key != '🏽' and key != '🏼' and key != '🏻': #filter color skin tone emojis
                uniqueEmojis.append(key)
    print("Number of filtered emojis: ", len(uniqueEmojis))
    return uniqueEmojis

"""
Read the emotional seed word corpus
"""
def getEmotionalSeedWords(config):
    with open(config['emotional_words_corpus_path']) as f:
        data = json.load(f)
    print("Emotional words: ", len(data.keys()))
    return data

"""
Get the emotional seed words of an interval given the interval vocabulary
"""
def getEmotionalSeedWordByInterval(emotional_words, voc):
    keys = emotional_words.keys()
    res = {}

    for key in keys:
        if key in voc:
            res[key] = emotional_words[key]

    print("Emotional words in the interval: ", len(res.keys()))
    return res

