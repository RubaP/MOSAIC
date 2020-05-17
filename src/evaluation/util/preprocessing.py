import collections
import datetime
import numpy as np
from src.components import trendIntervalDetection as timeInterval
from src.util import emotions as emotions, utils as util, textCleaner as textCleaner, textVocabulary as t_voc
from src.evaluation.Data import Data

"""
Initial point to load all the necessary data and create Data format
"""
def readAndProcessData(config):
    print("Config for Evaluation Data: ", config)

    emotioalSeedWordsAll = emotions.getEmotionalSeedWords(config['emotion'])
    emojiSeedWordsAll = emotions.get_emojiSeedWords(config['emotion'])

    df = util.readAndProcessFile(config)
    intervals = timeInterval.getIntervalsFromDataset(df, config['trend_interval_detection'])
    df = util.removeRetweet(df)
    df = textCleaner.clean(df)
    print("Preprocessing Finished: ", datetime.datetime.now())

    MEM = []
    EvaluationData = []

    print("=====Preprocessing intervals=====")
    for interval in intervals:
        print("-----------------------------------")
        print("Interval: ", interval)
        data = df[(df.timestamp >= interval[0]) & (df.timestamp < interval[1])]
        data = textCleaner.filterSmallDocs(data.copy())
        D_size = data.shape[0]
        print("data size: ", D_size)

        lda_data = {}
        emojiList = emotions.getEmojiList(data['emojis'])
        hashtags, hashtag_counter = textCleaner.getHashtagList(data['hashtags'])
        textual_vocabulary = t_voc.createTextVocabulary(data['cleanedText'], emojiList, hashtags)

        lda_data["data"] = data
        lda_data["D_size"] = D_size
        lda_data["emojiList"] = emojiList
        lda_data["textual_vocabulary"] = textual_vocabulary
        lda_data["textual_vocabulary_without_emojis"] = t_voc.getFilteredTextualVocabulary(textual_vocabulary,
                                                                                           emojiList)
        lda_data["emotionalSeedWords"] = emotions.getEmotionalSeedWordByInterval(emotioalSeedWordsAll, textual_vocabulary)
        lda_data["emojiSeedWordsAll"] = emojiSeedWordsAll
        lda_data['users'], lda_data['no_of_users'] = getUserList(data)
        loc_index, no_of_loc = cleanLocations(data)
        lda_data['loc_size'] = no_of_loc
        lda_data['locations'] = loc_index #for some baseline TE model
        lda_data['hashtag_counter'] =hashtag_counter
        eData = Data(config['topics'], data, textual_vocabulary)
        EvaluationData.append(eData)
        MEM.append(lda_data)

    MEM = findGenericSeedWords(MEM, config)
    return MEM, EvaluationData


def getUserList(data):
    users = data['user'].tolist()
    unique = list(set(users))

    res = []
    for user in users:
        res.append(unique.index(user))

    return res, len(unique)

"""
Convert location to index
"""
def cleanLocations(data):
    locations = data['location'].tolist()
    processed = [i for i in locations if i is not None and type(i) == str]
    tructed = [i.split(", ")[0] for i in processed]

    unique_lication = []
    count_by_loc = collections.Counter(tructed)

    for loc in count_by_loc.keys():
        if count_by_loc[loc] < 10:
            unique_lication.append(loc)
    no_of_uniqe_loc = len(unique_lication)
    print("Unique locatios: ", no_of_uniqe_loc)

    location_index = []

    for loc in locations:
        if loc is None or type(loc) != str:
            location_index.append(no_of_uniqe_loc) # last index is allocated for None
        else:
            strng = loc.split(" ")[0]
            if strng in unique_lication:
                location_index.append(unique_lication.index(strng))
            else:
                location_index.append(no_of_uniqe_loc)
    return location_index, no_of_uniqe_loc + 1

"""
Find the generic seed words using frequent hastags. Only the hastags are considered as they generally convey the generic information
"""
def findGenericSeedWords(MEM, config):
    common = None

    for data in MEM:
        if common is None:
            common = set(data['textual_vocabulary'])
        else:
            common = common.intersection(data['textual_vocabulary'])
    print("Common Words Size in Vocabulary: ", len(common))

    common_word_prob = {}
    for term in common:
        global_count = []
        for i in range(len(MEM)):
            local_count = 0
            lda_data = MEM[i]
            hashtag_counter = lda_data['hashtag_counter']

            if term in hashtag_counter:
                local_count += hashtag_counter[term]

            global_count.append(local_count)
        common_word_prob[term] = np.sum(global_count)

    genericSeedWords = []
    common_generic_print = []

    sorted_items = sorted(common_word_prob.items(), key=lambda x: x[1], reverse=True)

    for (term, count) in sorted_items:
        if len(genericSeedWords) < config['common_generic_threshold']:
            genericSeedWords.append(term)
            common_generic_print.append((term, count))

    common_generic_print.sort(key=lambda x: x[1], reverse=True)
    print("Common generic seed words (", len(genericSeedWords), "): ", common_generic_print)
    for i in range(len(MEM)):
        MEM[i]['genericSeedWords'] = genericSeedWords
    return MEM

"""
Get vector representation of words in the vocabulary
"""
def getWord2VecByVoc(voc, word2VecModel):
    res = {}
    notincount = 0
    missing_words = []
    for word in voc:
        try:
            res[word] = word2VecModel[word]
        except Exception as e:
            res[word] = None
            notincount +=1
            missing_words.append(word)
    print("Words not in embedding count ", notincount, " out of ", len(voc)) #because embedding has words which occur min 5 times
    return res