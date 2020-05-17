import src.evaluation.util.utils as utils
import numpy as np
import collections
from collections import Counter

class Data:
    def __init__(self, K, data, textual_vocabulary):
        self.K = K
        self.data = data.reset_index().copy()
        self.textual_vocabulary = textual_vocabulary
        self.T_size = len(textual_vocabulary)
        self.no_of_doc = data.shape[0]
        self.Words = [] #indices
        self.WordsT = [] #list of textual words
        self.W_bow = [] #vector

        for i, row in self.data.iterrows():
            words_list = []
            words_listT = []

            # Do word type assignment
            text = row['cleanedText']
            for t_word in text:
                if t_word in self.textual_vocabulary:
                    t = self.textual_vocabulary.index(t_word)
                    words_list.append(t)
                    words_listT.append(t_word)

            hashtags = row['hashtags']
            for t_word in hashtags:
                if t_word in self.textual_vocabulary:
                    t = self.textual_vocabulary.index(t_word)
                    words_list.append(t)
                    words_listT.append(t_word)

            emojis = row['emojis']
            for t_word in emojis:
                if t_word in self.textual_vocabulary:
                    t = self.textual_vocabulary.index(t_word)
                    words_list.append(t)
                    words_listT.append(t_word)

            self.Words.append(words_list)
            self.WordsT.append(words_listT)

    def setVoc(self, voc):
        self.textual_vocabulary = voc

    def getWBOW(self):
        W_bow = []
        for i in range(self.no_of_doc):
            W_bow.append(utils.generate_vector(self.T_size, self.Words[i]))
        return W_bow

    def setK(self, K):
        self.K = K

    def setZ(self, Z):
        self.Z = Z
        print("Popularity of documents: ", Counter(Z))

    def setphiE(self, phiE):
        self.phiE = phiE

    def setPhiT(self, dis):
        self.phiT = dis

    def setphiTE(self, phiTE):
        self.phiTE = phiTE

    def setphiTG(self, phiTG):
        self.phiTG = phiTG

    def getT_bow(self):
        T_bow = []

        for i in range(self.no_of_doc):
            t_words = np.array(self.Words[i])
            T_bow.append(utils.generate_vector(self.T_size, t_words))
        return T_bow