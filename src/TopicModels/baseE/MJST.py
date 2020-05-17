import numpy as np
from random import choices
import random

"""
Extension of JST to support emojis as a different modality
"""
class MJST:
    def __init__(self, id, K, lda_data, config):
        print("-----STARTED LDA INITIALIZATION-----")
        print("configuration: ", config)
        data = lda_data["data"].reset_index().copy()
        self.id = id
        self.interval = id
        self.K = K
        self.textual_vocabulary = lda_data["textual_vocabulary_without_emojis"]
        self.emoji_vocabulary = lda_data["emojiList"]
        self.config = config
        self.T_size = len(self.textual_vocabulary)
        print("Filtered textual voc size: ", self.T_size)
        self.P_size = len(self.emoji_vocabulary) #emojis are handled using the symbol P
        emotionalSeedWords = lda_data["emotionalSeedWords"]
        self.no_of_emotions = 7 #including neutral
        self.no_of_doc = data.shape[0]
        self.beta_E = np.append(self.config['beta_E'], 1)  # neutral is not available in config
        print("Number of topic: ", K)

        #Textual words
        self.Words = []
        self.Emojis = []

        #Z and E assignment of words and emojis
        self.ZW = []
        self.ZP = []
        self.EW = []
        self.EP = []

        #Count matrices
        self.C_DE = np.zeros((self.no_of_doc, self.no_of_emotions)) #N_kd
        self.C_DKE = np.zeros((self.no_of_doc, K, self.no_of_emotions)) #N_jkd

        self.C_WKE_total = np.zeros((K, self.no_of_emotions)) #N_jk
        self.C_PKE_total = np.zeros((K, self.no_of_emotions))  # N_jk

        self.C_WKE = np.zeros((K, self.no_of_emotions, self.T_size))  # N_jkl
        self.C_PKE = np.zeros((K, self.no_of_emotions, self.P_size))  # N_jkl

        self.stabilization =[]

        # To obtain necessary distributions
        self.phiTSum = np.zeros((self.K, self.T_size + self.P_size))
        self.phiTESum = np.zeros((self.no_of_emotions, self.T_size + self.P_size))
        self.phiESum = np.zeros((K, self.no_of_emotions))
        self.updateParamCount = 0
        self.Z_prob = np.zeros((self.no_of_doc, self.K))
        self.C_DK = np.zeros((self.no_of_doc, K)) #NOTE, DK does not include emoticons
        self.lambda_TS = config['lambda_TS']

        for i, row in data.iterrows():
            words_list = []
            emoji_list = []
            z_listW = []
            e_listW = []
            z_listP = []
            e_listP = []

            text = row['cleanedText']
            for t_word in text:
                if t_word in self.textual_vocabulary:
                    t = self.textual_vocabulary.index(t_word)
                    topic = np.random.randint(K)

                    if t_word in emotionalSeedWords.keys():
                        emotion = random.choice(emotionalSeedWords[t_word])
                    elif t_word in emotionalSeedWords.keys():
                        emotion_prob = emotionalSeedWords[t_word]
                        emotion = choices(range(7), weights=list(map(float, emotion_prob)))[0]
                    else:
                        emotion = choices(range(7), np.random.dirichlet(self.beta_E, 1)[0, :])[0]

                    self.C_DE[i][emotion] +=1
                    self.C_DKE[i][topic][emotion] +=1
                    self.C_WKE[topic][emotion][t] +=1
                    self.C_WKE_total[topic][emotion] +=1
                    self.C_DK[i][topic] += 1

                    z_listW.append(topic)
                    e_listW.append(emotion)
                    words_list.append(t)
            self.ZW.append(z_listW)
            self.EW.append(e_listW)
            self.Words.append(words_list)

            emojis = row['emojis']
            for t_word in emojis:
                if t_word in self.emoji_vocabulary:
                    t = self.emoji_vocabulary.index(t_word)
                    emotion = np.random.randint(self.no_of_emotions)
                    if t_word in lda_data['emojiSeedWordsAll'].keys():
                        emotion_prob = lda_data['emojiSeedWordsAll'][t_word]
                        emotion = choices(range(7), weights=list(map(float, emotion_prob)))[0]

                    self.C_DE[i][emotion] += 1
                    self.C_DKE[i][topic][emotion] += 1
                    self.C_PKE[topic][emotion][t] += 1
                    self.C_PKE_total[topic][emotion] += 1

                    z_listP.append(topic)
                    e_listP.append(emotion)
                    emoji_list.append(t)
            self.ZP.append(z_listP)
            self.EP.append(e_listP)
            self.Emojis.append(emoji_list)

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
                    old_emotion = self.EW[i][j]

                    self.C_DE[i][old_emotion] -= 1
                    self.C_DKE[i][old_topic][old_emotion] -= 1
                    self.C_WKE[old_topic][old_emotion][word] -= 1
                    self.C_WKE_total[old_topic][old_emotion] -= 1
                    self.C_DK[i][old_topic] -= 1

                    topic, emotion = self.find_topicAndEmotionW(i, word)
                    self.C_DE[i][emotion] += 1
                    self.C_DKE[i][topic][emotion] += 1
                    self.C_WKE[topic][emotion][word] += 1
                    self.C_WKE_total[topic][emotion] += 1
                    self.C_DK[i][topic] += 1

                    self.ZW[i][j] = topic
                    self.EW[i][j] = emotion

                no_of_Tword = len(self.Emojis[i])
                for j in range(no_of_Tword):
                    word = self.Emojis[i][j]
                    old_topic = self.ZP[i][j]
                    old_emotion = self.EP[i][j]

                    self.C_DE[i][old_emotion] -= 1
                    self.C_DKE[i][old_topic][old_emotion] -= 1
                    self.C_PKE[old_topic][old_emotion][word] -= 1
                    self.C_PKE_total[old_topic][old_emotion] -= 1

                    topic, emotion = self.find_topicAndEmotionP(i, word)
                    self.C_DE[i][emotion] += 1
                    self.C_DKE[i][topic][emotion] += 1
                    self.C_PKE[topic][emotion][word] += 1
                    self.C_PKE_total[topic][emotion] += 1

                    self.ZP[i][j] = topic
                    self.EP[i][j] = emotion

            if n % self.config['sample_lag'] == 0 and n > self.config['burn_in']:
                self.updateParams()

        self.updateParams()
        self.updateDistributions()

    def find_topicAndEmotionW(self, i, word):
        term1 = (self.C_WKE[:,:,word] + self.lambda_TS) / (self.C_WKE_total + self.T_size * self.lambda_TS) #n2/d2
        term2 = (self.C_DE[i] + self.beta_E)/(self.C_DE[i] + self.K*self.config['beta_Z']) #n3/d1
        term3 = np.multiply((self.C_DKE[i] + self.config['beta_Z']), term2) #n1


        p_z1 = np.multiply(term1, term3)
        p_z2 = p_z1.flatten()
        new = np.argmax(np.random.multinomial(1, p_z2 / p_z2.sum()))
        return int(new/self.no_of_emotions), new % self.no_of_emotions

    def find_topicAndEmotionP(self, i, word):
        term1 = (self.C_PKE[:,:,word] + 2)/(self.C_PKE_total + self.T_size * 2)
        term2 = (self.C_DE[i] + self.config['beta_Z'])/(self.C_DE[i] + self.K*self.config['beta_Z'])
        term3 = np.divide((self.C_DKE[i] + self.config['beta_Z']), term2)


        p_z1 = np.multiply(term1, term3)
        p_z2 = p_z1.flatten()
        new = np.argmax(np.random.multinomial(1, p_z2 / p_z2.sum()))
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
        common_WKE = np.concatenate((self.C_WKE, self.C_PKE), axis=2)

        # Texual word distribution per topic
        temp = common_WKE.sum(axis=1) + self.lambda_TS
        row_sum = temp.sum(axis=1)
        self.phiTSum += (temp / row_sum[:, np.newaxis])

        # Emotion distribution over per topic
        temp = common_WKE.sum(axis=2) + np.array(self.beta_E)
        row_sum = temp.sum(axis=1)
        self.phiESum += temp / row_sum[:, np.newaxis]

        # Texual word distribution over emotion
        temp = common_WKE.sum(axis=0) + self.config['lambda_TE']
        row_sum = temp.sum(axis=1)
        self.phiTESum += temp / row_sum[:, np.newaxis]

        self.Z_prob = np.add(self.Z_prob, self.C_DK)

        self.updateParamCount += 1