from more_itertools import locate
import numpy as np
from scipy.stats import entropy
from langdetect import detect
import src.util.textCleaner as textCleaner

"""
Generate event summary with representative microblog and emotion. Summary is sorted by time of the topic
"""
def generateSummary(EvaluationData, with_emotion=True):
    print("===========Started Content Selection=============")
    summary = []
    for id, eData in enumerate(EvaluationData):
        K = eData.K

        for k in range(K):
            indices = list(locate(eData.Z, lambda x: x == k))
            d = eData.data
            d = d.loc[indices]
            d = d[d.reply == False].copy()
            d["significance"] = d["followers_log"] + d["retweet_count_log"]
            significance_max = d["significance"].max()
            d["significance"] = d["significance"] / significance_max

            summary.append({"id": id,
                            "topicId": k,
                            "time": np.median(d['timestamp'].tolist()),
                            "e_dis": eData.phiE[k] if with_emotion else None,
                            "text": chooseText(d, eData.phiT[k], eData.textual_vocabulary),
                            "images": None,
                            "topic_words": getTopicWords(eData.phiT[k], eData.textual_vocabulary)
                            })
        print("---------------------------------------")

    print("==========Content Selection Completed===============")
    summary.sort(key=lambda x : x['time'])
    return summary

"""
Get top 5 topic words
"""
def getTopicWords(dis, voc):
    word_dis = np.array(dis)
    top5 = word_dis.argsort()[-5:][::-1]
    res = []
    for word in top5:
        res.append(voc[word])
    return res

"""
Chooses a single representative microblog, given a topic distribution and data
"""
def chooseText(data, t_dis, text_voc):
    word_dis = np.array(t_dis)  # bow is shorten for an efficient computation
    top = word_dis.argsort()[-5:][::-1].tolist()
    t_dis = word_dis[top]
    t_dis = t_dis / t_dis.sum()

    # compute bow, distribution and significance
    data = data.apply(lambda x: textCleaner.cleanDataForPresentation(x), axis=1)
    data["T_bow"] = data["cleanDataPresentation"].apply(lambda x: generate_filtered_bag_of_words(text_voc, x, top))

    data["top_word_count"] = data["T_bow"].apply(lambda x: x.sum())
    temp = data.copy()
    data = data[data.top_word_count >= 2].copy()
    if data.shape[0] == 0:
        data = temp

    data["T_dis"] = data["T_bow"].apply(lambda x: x / x.sum())
    data["coverage"] = data["T_dis"].apply(lambda x: entropy(x, t_dis))
    coverage_max = data["coverage"].max()
    data["coverage"] = data["coverage"] / coverage_max
    data["coverage"] = 1 - data["coverage"]
    data["coverage + significance"] = 0.1 * data["significance"] + 0.9 * data["coverage"]

    # choose initial data
    data_index = data["coverage + significance"].idxmax()
    initial_data = data.ix[data_index]
    print("Representative Microblog: ", initial_data['text'])
    return (initial_data['timestamp'], initial_data['text'])

"""
generate a shorten version of bag of words using a filtered word index voc top
"""
def generate_filtered_bag_of_words(voc, words, top, default=0.01):
    bow = np.array([default]*len(top))
    for word in words:
        try:
            indx = voc.index(word)
            top_index = top.index(indx)
            bow[top_index] = bow[top_index] + 1
        except:
            pass
    return bow
