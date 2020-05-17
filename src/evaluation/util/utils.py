from more_itertools import locate
import numpy as np
from scipy.stats import entropy
from itertools import chain
from sklearn.cluster import MiniBatchKMeans
import datetime
import warnings
import copy
import src.util.textCleaner as textCleaner
from collections import Counter

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec

def generate_vector(voc_size, words):
    bow = np.array([0]*voc_size)
    for index in words:
        index = int(index)
        bow[index] = bow[index] + 1
    return bow

def generate_bag_of_words(text_voc, words, default=0.01):
    bow = np.array([default]*len(text_voc))
    for word in words:
        try:
            index = text_voc.index(word)
            bow[index] = bow[index] + 1
        except:
            pass
    return bow

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

def getDocumentForm(no_of_doc, labels, data):
    bow = data.getWBOW()
    docs = np.zeros((no_of_doc, data.T_size))

    for i in range(len(labels)):
        docs[labels[i]] += bow[i]
    print("Docs: ", docs.shape)
    return docs

def getAggregatedBOW(no_of_doc, labels, data):
    bow = data.Words
    docs = [None]*no_of_doc

    for i in range(len(labels)):
        try:
            if docs[labels[i]] == None:
                docs[labels[i]] = copy.deepcopy(bow[i])
            else:
                docs[labels[i]].extend(copy.deepcopy(bow[i]))
        except:
            print(len(labels), len(bow))
            exit()
    return [i for i in docs if i is not None] #filter None

def getAggregatedWords(no_of_doc, labels, data):
    words = data.WordsT
    docs = [None]*no_of_doc

    for i in range(len(labels)):
        if docs[labels[i]] == None:
            docs[labels[i]] = copy.deepcopy(words[i])
        else:
            docs[labels[i]].extend(copy.deepcopy(words[i]))
    return [i for i in docs if i is not None] #filter None


def getDocsGroupByHashtags(EvaluationData):
    print("Started grouping by hashtag: ", datetime.datetime.now())
    docs = []
    doc_sizes = []

    for data in EvaluationData:
        hashtags = data.data['hashtags'].tolist()
        voc = list(set(list(chain.from_iterable(hashtags))))
        print("No of hashtags: ", len(voc))
        hashtag_vec = [generate_bag_of_words(voc, word, default=0) for word in hashtags]
        non_zero = np.count_nonzero(hashtag_vec)
        print("Nonzero entry: ", non_zero)
        doc_size = int(data.no_of_doc * len(voc) / non_zero)
        print("Doc size: ", doc_size)
        kmeans = MiniBatchKMeans(n_clusters=doc_size).fit(hashtag_vec)
        docs.append(getDocumentForm(doc_size, kmeans.labels_, data))
        doc_sizes.append(doc_size)
    print("Finished grouping by hashtag: ", datetime.datetime.now())
    return docs, doc_sizes


def sent_vectorizer(sent, model):
    sent_vec = []
    numw = 0
    for w in sent:
        try:
            if numw == 0:
                sent_vec = model[w]
            else:
                sent_vec = np.add(sent_vec, model[w])
            numw += 1
        except:
            pass

    return np.asarray(sent_vec) / numw

def getDocsGroupByEmbedding(EvaluationData, doc_sizes):
    print("Started grouping by WMD: ", datetime.datetime.now())

    docs = []
    words = []
    count = 0
    for data in EvaluationData:
        text = data.data['cleanedCompleteText'].tolist()
        model = Word2Vec(text, min_count=1)

        X = []
        for sentence in text:
            X.append(sent_vectorizer(sentence, model))

        kmeans = MiniBatchKMeans(n_clusters=doc_sizes[count]).fit(X)
        docs.append(getAggregatedBOW(doc_sizes[count], kmeans.labels_, data))
        words.append(getAggregatedWords(doc_sizes[count], kmeans.labels_, data))
        count +=1

    print("Finished grouping by embedding: ", datetime.datetime.now())
    return docs, getEdges(words)

def getEdges(docs_list):
    print("Started creating edges: ", datetime.datetime.now())
    edges_list = []

    for docs in docs_list:
        model = Word2Vec(docs, min_count=1)
        edges_by_int = []
        for m in range(len(docs)):
            edges_by_doc = []  # doc
            words = docs[m]
            for i in range(len(words)):
                edges = []
                for j in range(len(words)):
                    if i != j:
                        sim = model.wv.similarity(w1=words[i], w2=words[j])
                        if sim > 0.7:
                            edges.append(j)
                edges_by_doc.append(edges)
            edges_by_int.append(edges_by_doc)
        edges_list.append(edges_by_int)

    print("Finished creating edges: ", datetime.datetime.now())
    return edges_list

def getZFromZW(Z, no_of_doc):
    res = np.array([0]*no_of_doc)
    for i in range(no_of_doc):
        if len(Z[i]) > 0:
            res[i] = Counter(Z[i]).most_common()[0][0]
    return res
