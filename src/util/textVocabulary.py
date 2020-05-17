from nltk import FreqDist

"""
Filter the vocabulary given a filtering list e.g. excluding emoticons
"""
def getFilteredTextualVocabulary(voc, filtering_list):
    return [word for word in voc if word not in filtering_list]

"""
Create vocabulary for each interval
"""
def createTextVocabulary(texts, emojis, hashtags):
    print("-----STARTED CREATING TEXT VOCABULARY----")
    word_dist = FreqDist()
    texts.apply(lambda x: "" if x is None else word_dist.update(x))

    voc = dict(word_dist)

    for k, v in list(voc.items()):
        if v < 5:
            del voc[k]
    voc = list(voc.keys())

    voc.extend(hashtags)
    voc = list(set(voc))

    voc.sort()
    filtered_voc = voc

    filtered_voc.extend(emojis)  # Emoji is extended last to support multiple vocabulary
    print("Texual Vocabulary created with the size: ", len(filtered_voc))
    print("-----FINISHED TEXTUAL VOCABULARY CREATION-----")
    return filtered_voc