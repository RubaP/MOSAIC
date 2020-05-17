import numpy as np
from random import choices
import random
"""
Latest one, which supports bias (objective and subjective)
"""
class BST:
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
        self.lambda_TS = config['lambda_TS']
        self.beta_B = config['beta_T'][0]

        print("Number of topic: ", K)

        #Textual words
        self.Words = Words

        self.ZW = []
        self.E = []
        self.B = []

        #Count matrices
        self.C_DB = np.zeros((self.no_of_doc, 2))
        self.C_DBE = np.zeros((self.no_of_doc, 2, self.no_of_emotions))  #
        self.C_DBEK = np.zeros((self.no_of_doc, 2, self.no_of_emotions, K))  #
        self.C_BEKW = np.zeros((2, self.no_of_emotions, K, self.T_size))  #
        self.C_BEKW_total = np.zeros((2, self.no_of_emotions, K))  #

        #To obtain necessary distributions
        self.phiTSum = np.zeros((self.K, self.T_size))
        self.phiTESum = np.zeros((self.no_of_emotions, self.T_size))
        self.phiESum = np.zeros((K, self.no_of_emotions))
        self.updateParamCount = 0
        self.Z_prob = np.zeros((self.no_of_doc, self.K))
        self.C_DK = np.zeros((self.no_of_doc, K))

        for i in range(self.no_of_doc):
            words_list = Words[i]
            z_list = []
            e_list = []
            b_list = []

            for t in words_list:
                t_word = self.textual_vocabulary[t]
                topic = np.random.randint(K)
                c = choices(range(2), np.random.dirichlet([self.beta_B]*2, 1)[0, :])[0]
                if t_word in emotionalSeedWords.keys():
                    c = 1
                    emotion = random.choice(emotionalSeedWords[t_word])
                elif t_word in lda_data['emojiSeedWordsAll'].keys():
                    c = 1
                    emotion_prob = lda_data['emojiSeedWordsAll'][t_word]
                    emotion = choices(range(7), weights=list(map(float, emotion_prob)))[0]
                else:
                    emotion = choices(range(7), np.random.dirichlet(self.beta_E, 1)[0, :])[0]

                self.C_DB[i][c] +=1
                self.C_DBE[i][c][emotion] += 1
                self.C_DBEK[i][c][emotion][topic] +=1
                self.C_BEKW[c][emotion][topic][t] +=1
                self.C_BEKW_total[c][emotion][topic] +=1
                self.C_DK[i][topic] += 1

                z_list.append(topic)
                e_list.append(emotion)
                b_list.append(c)

            self.ZW.append(z_list)
            self.E.append(e_list)
            self.B.append(b_list)

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
                    old_b = self.B[i][j]

                    self.C_DB[i][old_b] -= 1
                    self.C_DBE[i][old_b][old_emotion] -= 1
                    self.C_DBEK[i][old_b][old_emotion][old_topic] -= 1
                    self.C_BEKW[old_b][old_emotion][old_topic][word] -= 1
                    self.C_BEKW_total[old_b][old_emotion][old_topic] -=1
                    self.C_DK[i][old_topic] -= 1

                    bias, topic, emotion = self.find_biasTopicAndEmotion(i, word)
                    self.C_DB[i][bias] += 1
                    self.C_DBE[i][bias][emotion] += 1
                    self.C_DBEK[i][bias][emotion][topic] += 1
                    self.C_BEKW[bias][emotion][topic][word] += 1
                    self.C_BEKW_total[bias][emotion][topic] += 1
                    self.C_DK[i][topic] += 1

                    self.ZW[i][j] = topic
                    self.E[i][j] = emotion
                    self.B[i][j] = bias

            if n % self.config['sample_lag'] == 0 and n > self.config['burn_in']:
                self.updateParams()

        self.updateParams()
        self.updateDistributions()

    def find_biasTopicAndEmotion(self, i, word):
        term1 = (self.C_DB[i] + self.beta_B)/(self.C_DB[i] + 2*self.beta_B)
        term2 = (self.C_DBE[i] + self.beta_E)/(self.C_DBE[i] + self.beta_ESum)
        term3 = (self.C_DBEK[i] + self.config['beta_Z'])
        term4 = (self.C_BEKW[:,:,:,word] + self.lambda_TS) / (self.C_BEKW_total + self.T_size * self.lambda_TS)

        p_z1 = term2*term1[:, np.newaxis]
        p_z2 = term3*p_z1[:, :, np.newaxis]
        p_z3 = p_z2*term4

        p_z4 = p_z3.flatten()
        index = np.argmax(np.random.multinomial(1, p_z4 / p_z4.sum()))

        newc = int(index % (self.K * self.no_of_emotions * 2) / (self.K * self.no_of_emotions))
        news = int(index % (self.K * self.no_of_emotions) / self.K)
        newz = index % self.K
        return newc, newz, news

    def updateDistributions(self):
        self.Z = np.argmax(self.Z_prob, axis=1)
        self.phiT = self.phiTSum / self.updateParamCount
        self.phiTE = self.phiTESum / self.updateParamCount
        self.phiE = self.phiESum / self.updateParamCount
        print("====================================")

        print("Topic-Emotion dis: ")
        print(self.phiE)

    def updateParams(self):
        C_WKE = np.swapaxes(self.C_BEKW.sum(axis=0), 0, 1)
        # Texual word distribution per topic
        temp = C_WKE.sum(axis=1) + self.lambda_TS
        row_sum = temp.sum(axis=1)
        self.phiTSum += (temp / row_sum[:, np.newaxis])

        # Emotion distribution over per topic
        temp = C_WKE.sum(axis=2) + np.array(self.beta_E)
        row_sum = temp.sum(axis=1)
        self.phiESum += temp / row_sum[:, np.newaxis]

        # Texual word distribution over emotion
        temp = C_WKE.sum(axis=0) + self.config['lambda_TE']
        row_sum = temp.sum(axis=1)
        self.phiTESum += temp / row_sum[:, np.newaxis]

        self.Z_prob = np.add(self.Z_prob,self.C_DK)

        self.updateParamCount += 1