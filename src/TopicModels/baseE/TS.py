import numpy as np
from random import choices
import random
from collections import Counter
"""
Extension of LDA to support emotion
"""
class TS:
    def __init__(self, id, K, lda_data, config, Words):
        print("-----STARTED LDA INITIALIZATION-----")
        print("configuration: ", config)
        self.id = id
        self.interval = id
        self.K = K
        self.textual_vocabulary = lda_data["textual_vocabulary"]
        self.config = config
        self.T_size = len(lda_data["textual_vocabulary"])
        emotionalSeedWords = lda_data["emotionalSeedWords"]
        self.no_of_emotions = 7 #including neutral
        self.no_of_doc = len(Words)
        self.beta_E = np.append(self.config['beta_E'], 1)  # neutral is not available in config
        self.beta_ESum = self.beta_E.sum()
        print("Number of topic: ", K)

        #Textual words
        self.Words = Words

        self.ZW = []
        self.E = []

        #Count matrices
        self.C_KE = np.zeros((K, self.no_of_emotions)) #N_jk
        self.C_WKE = np.zeros((K, self.no_of_emotions, self.T_size)) #N_jik
        self.C_DK = np.zeros((self.no_of_doc, K)) #N_dj

        #Total value matrices
        self.C_WK_total = np.zeros(K) #N_j

        self.stabilization =[]

        # To obtain necessary distributions
        self.phiTSum = np.zeros((self.K, self.T_size))
        self.phiTESum = np.zeros((self.no_of_emotions, self.T_size))
        self.phiESum = np.zeros((K, self.no_of_emotions))
        self.updateParamCount = 0
        self.Z_prob = np.zeros((self.no_of_doc, self.K))

        for i in range(self.no_of_doc):
            words_list = Words[i]
            z_list = []
            e_list = []

            for t in words_list:
                t_word = self.textual_vocabulary[t]
                topic = np.random.randint(K)

                if t_word in emotionalSeedWords.keys():
                    emotion = random.choice(emotionalSeedWords[t_word])
                elif t_word in lda_data['emojiSeedWordsAll'].keys():
                    emotion_prob = lda_data['emojiSeedWordsAll'][t_word]
                    emotion = choices(range(7), weights=list(map(float, emotion_prob)))[0]
                else:
                    emotion = choices(range(7), np.random.dirichlet(self.beta_E, 1)[0, :])[0]

                self.C_KE[topic][emotion] +=1
                self.C_WKE[topic][emotion][t] +=1
                self.C_DK[i][topic] +=1
                self.C_WK_total[topic] +=1

                z_list.append(topic)
                e_list.append(emotion)
            self.ZW.append(z_list)
            self.E.append(e_list)

        print("+++++INITIALIZATION COMPLETED+++++")

    def inference(self):
        iterations = self.config["maximum_iteration"]

        print("++++++START SAMPLING+++++++++")
        print("No of documents: ", self.no_of_doc)
        for n in range(iterations):
            for i in range(self.no_of_doc):
                no_of_Tword = len(self.Words[i])
                for j in range(no_of_Tword):
                    word = self.Words[i][j]
                    old_topic = self.ZW[i][j]
                    old_emotion = self.E[i][j]

                    self.C_KE[old_topic][old_emotion] -= 1
                    self.C_WKE[old_topic][old_emotion][word] -= 1
                    self.C_DK[i][old_topic] -= 1
                    self.C_WK_total[old_topic] -= 1

                    topic, emotion = self.find_topicAndEmotion(i, word)
                    self.C_KE[topic][emotion] += 1
                    self.C_WKE[topic][emotion][word] += 1
                    self.C_DK[i][topic] += 1
                    self.C_WK_total[topic] += 1
                    self.ZW[i][j] = topic
                    self.E[i][j] = emotion

            if n % self.config['sample_lag'] == 0 and n > self.config['burn_in']:
                self.updateParams()

        self.updateParams()
        self.updateDistributions()

    def find_topicAndEmotion(self, i, word):
        term1 = (self.C_DK[i] + self.config['beta_Z'])/(self.C_WK_total + self.beta_ESum) #n1/d3
        term2 = (self.C_KE + self.beta_E)/((self.C_KE + self.T_size*self.config['lambda_TS'])) #n3/d2
        term3 = self.C_WKE[:,:,word] + self.config['lambda_TS'] #n2


        p_z1 = np.multiply(term2, term3)
        p_z2 = np.multiply(p_z1, term1[:,np.newaxis])
        p_z3 = p_z2.flatten()
        new = np.argmax(np.random.multinomial(1, p_z3 / p_z3.sum()))
        return int(new/self.no_of_emotions), new % self.no_of_emotions

    def updateDistributions(self):
        self.Z = np.argmax(self.Z_prob, axis=1)
        self.phiT = self.phiTSum / self.updateParamCount
        self.phiTE = self.phiTESum / self.updateParamCount
        self.phiE = self.phiESum / self.updateParamCount
        print("====================================")

        print("Topic-Emotion dis: ")
        print(self.phiE)

    def updateParams(self):

        # Texual word distribution per topic
        temp = self.C_WKE.sum(axis=1) + self.config['lambda_TS']
        row_sum = temp.sum(axis=1)
        self.phiTSum += (temp / row_sum[:, np.newaxis])

        # Emotion distribution over per topic
        temp = self.C_WKE.sum(axis=2) + np.array(self.beta_E)
        row_sum = temp.sum(axis=1)
        self.phiESum += temp / row_sum[:, np.newaxis]

        # Texual word distribution over emotion
        temp = self.C_WKE.sum(axis=0) + self.config['lambda_TE']
        row_sum = temp.sum(axis=1)
        self.phiTESum += temp / row_sum[:, np.newaxis]

        self.Z_prob = np.add(self.Z_prob, self.C_DK)

        self.updateParamCount += 1