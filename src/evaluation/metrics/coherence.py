import math
import numpy as np

"""
Global word counts are maintained to speedup the process
"""
global_word_count = None
global_word_coocurence = None

def initializeGlobalWordCount(lda):
    global global_word_count
    global global_word_coocurence

    if global_word_count is None:
        global_word_count = [{}]*len(lda)
        global_word_coocurence = [{}]*len(lda)

def getGlobalWordCounts(interval, word1, word2):
    global global_word_count
    global global_word_coocurence
    if word1 < word2:
        if word1 in global_word_coocurence[interval] and word2 in global_word_coocurence[interval][word1]:
            count_cooc = global_word_coocurence[interval][word1][word2]
            count_ind = global_word_count[interval][word2]
            count_indm = global_word_count[interval][word1]
        else:
            return None, None, None
    elif word2 in global_word_coocurence[interval] and word1 in global_word_coocurence[interval][word2]:
        count_cooc = global_word_coocurence[interval][word2][word1]
        count_ind = global_word_count[interval][word2]
        count_indm = global_word_count[interval][word1]
    else:
        return None, None, None
    return count_indm, count_ind, count_cooc

def updateGlobalWordCount(interval, word1, word2, word1Count, word2Count, cooccurence_count):
    global global_word_coocurence
    global global_word_count

    if word1 not in global_word_count[interval]:
        global_word_count[interval][word1] = word1Count
    if word2 not in global_word_count[interval]:
        global_word_count[interval][word2] = word2Count

    if word1 < word2:
        if word1 in global_word_coocurence[interval]:
            global_word_coocurence[interval][word1][word2] = cooccurence_count
        else:
            global_word_coocurence[interval][word1] = {word2: cooccurence_count}
    else:
        if word2 in global_word_coocurence[interval]:
            global_word_coocurence[interval][word2][word1] = cooccurence_count
        else:
            global_word_coocurence[interval][word2] = {word1: cooccurence_count}

def calcluateCoherence(Doc, Words, no_of_doc, interval):
    if len(Words) == 0:
        return None
    PMI5 = 0
    PMI10 = 0
    PMI20 = 0

    for m in range(1,len(Words)):
        for l in range(m):
            count_indm, count_ind, count_cooc = getGlobalWordCounts(interval, Words[m], Words[l])
            if count_cooc is None:
                count_cooc =0
                count_ind = 0
                count_indm = 0 #for PMI
                for doc in Doc:
                    if doc == None or len(doc) == 0:
                        continue
                    elif Words[l] in doc:
                        count_ind +=1
                        if Words[m] in doc:
                            count_indm += 1
                            count_cooc +=1
                    elif Words[m] in doc:
                        count_indm +=1
                updateGlobalWordCount(interval, Words[m], Words[l], count_indm, count_ind, count_cooc)

            PMI = math.log((count_cooc + 1 )*no_of_doc/(count_ind*count_indm))

            if m < 5:
                PMI5 += PMI
            if m < 10:
                PMI10 += PMI
            PMI20 += PMI

    const5 = 2/(5*4)
    const10 = 2 / (10 * 9)
    const20 = 2 / (20 * 19)
    return [const5*PMI5, const10*PMI10, const20*PMI20]


"""
Compute coherence of K topic dis PhiT
"""
def compute_coherence(lda_list):
    initializeGlobalWordCount(lda_list)
    PMI_listT5 = []
    PMI_listT10 = []
    PMI_listT20 = []

    intervals = len(lda_list)
    for interval in range(intervals):
        lda = lda_list[interval]
        TWords = lda.Words
        no_of_doc = len(TWords)

        print("=======================")
        for k in range(lda.K):
            word_dis = np.array(lda.phiT[k])
            top20 = word_dis.argsort()[-20:][::-1]

            try:
                strng = "Topic: " + str(k) + " : "
                for word in top20:
                    strng += lda.textual_vocabulary[word] + "[" + str(word) + ":" + str(
                        format(word_dis[word], '.2f')) + "]" + ", "
                print(strng)
            except:
                pass

            PMI = calcluateCoherence(TWords, top20, no_of_doc, interval)
            PMI_listT5.append(PMI[0])
            PMI_listT10.append(PMI[1])
            PMI_listT20.append(PMI[2])
        print("=======================")

    print("PMI5", np.mean(PMI_listT5))
    print("PMI10", np.mean(PMI_listT10))
    print("PMI20", np.mean(PMI_listT20))

    return [np.mean(PMI_listT5), np.mean(PMI_listT10), np.mean(PMI_listT20)]