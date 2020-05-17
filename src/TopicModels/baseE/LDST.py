import numpy as np
from random import choices
import random
import scipy
from collections import Counter

class LDST:
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
        self.no_of_emotions = 7
        self.no_of_doc = len(Words)
        self.L_size = lda_data['loc_size']
        self.U_size = lda_data['no_of_users']
        self.U = lda_data['users']
        timestamp = np.array(lda_data['data']['timestamp'].tolist())
        self.timestamp = (timestamp - min(timestamp))/(max(timestamp) - min(timestamp))
        self.beta_E = np.append(self.config['beta_E'], 1)  # neutral is not available in config
        self.lambda_TS = config['lambda_TS']
        self.beta_Z = config['beta_Z']
        print("Number of topic: ", K)

        #Textual words
        self.Words = Words

        self.C_UC = np.zeros((self.U_size, 2))
        self.C_KEW = np.zeros((K, self.no_of_emotions, self.T_size))
        self.C_KEW_total = np.zeros((K, self.no_of_emotions))

        self.C_LE = np.zeros((self.L_size, self.no_of_emotions))
        self.C_LE_total = np.zeros(self.L_size)
        self.C_UE = np.zeros((self.U_size, self.no_of_emotions))
        self.C_UE_total = np.zeros(self.U_size)

        self.C_UET = np.zeros((self.U_size, K, self.no_of_emotions))
        self.C_UET_total = np.zeros((self.U_size, self.no_of_emotions))
        self.C_LET = np.zeros((self.L_size, K, self.no_of_emotions))
        self.C_LET_total = np.zeros((self.L_size, self.no_of_emotions))

        self.Psi = np.array([[1 for _ in range(2)] for _ in range(K)])
        self.Betafunc_Psi = [scipy.special.beta(self.Psi[z][0], self.Psi[z][1]) for z in range(K)]
        self.DZ = np.zeros(self.no_of_doc)

        # To obtain necessary distributions
        self.phiTSum = np.zeros((self.K, self.T_size))
        self.phiTESum = np.zeros((self.no_of_emotions, self.T_size))
        self.phiESum = np.zeros((K, self.no_of_emotions))
        self.updateParamCount = 0
        self.Z_prob = np.zeros((self.no_of_doc, self.K))
        self.C_DK = np.zeros((self.no_of_doc, K))

        self.L = lda_data['locations']
        self.ZW = []
        self.E = []
        self.Etype = []

        for i in range(self.no_of_doc):
            location = self.L[i]
            user = self.U[i]
            words_list = Words[i]
            z_list = []
            e_list = []
            etype_list = []

            for t in words_list:
                t_word = self.textual_vocabulary[t]
                z = np.random.randint(K)
                c = int(random.random() * 2)

                if t_word in emotionalSeedWords.keys():
                    emotion = random.choice(emotionalSeedWords[t_word])
                elif t_word in lda_data['emojiSeedWordsAll'].keys():
                    emotion_prob = lda_data['emojiSeedWordsAll'][t_word]
                    emotion = choices(range(7), weights=list(map(float, emotion_prob)))[0]
                else:
                    emotion = choices(range(7), np.random.dirichlet(self.beta_E, 1)[0, :])[0]

                self.C_UC[user][c] += 1
                self.C_KEW[z][emotion][t] += 1
                self.C_KEW_total[z][emotion] += 1
                self.C_DK[i][z] += 1

                if c == 0:
                    self.C_UE[user][emotion] += 1
                    self.C_UE_total[user] += 1
                    self.C_UET[user][z][emotion] += 1
                    self.C_UET_total[user][emotion] += 1
                else:
                    self.C_LE[location][emotion] += 1
                    self.C_LE_total[location] += 1
                    self.C_LET[location][z][emotion] += 1
                    self.C_LET_total[location][emotion] += 1

                z_list.append(z)
                e_list.append(emotion)
                etype_list.append(c)
            self.ZW.append(z_list)
            if len(z_list) > 0:
                self.DZ[i] = Counter(z_list).most_common()[0][0]
            self.E.append(e_list)
            self.Etype.append(etype_list)

    print("+++++INITIALIZATION COMPLETED+++++")

    def inference(self):
        iterations = self.config["maximum_iteration"]

        print("++++++START SAMPLING+++++++++")
        print("No of documents: ", self.no_of_doc)
        for n in range(iterations):
            for i in range(self.no_of_doc):
                location = self.L[i]
                user = self.U[i]
                no_of_Tword = len(self.Words[i])
                for j in range(no_of_Tword):
                    word = self.Words[i][j]
                    old_topic = self.ZW[i][j]
                    old_emotion = self.E[i][j]
                    old_etype = self.Etype[i][j]

                    self.C_UC[user][old_etype] -= 1
                    self.C_KEW[old_topic][old_emotion][word] -= 1
                    self.C_KEW_total[old_topic][old_emotion] -= 1
                    self.C_DK[i][old_topic] -= 1

                    if old_etype == 0:
                        self.C_UE[user][old_emotion] -= 1
                        self.C_UE_total[user] += 1
                        self.C_UET[user][old_topic][old_emotion] -= 1
                        self.C_UET_total[user][old_emotion] -= 1
                    else:
                        self.C_LE[location][old_emotion] -= 1
                        self.C_LE_total[location] -= 1
                        self.C_LET[location][old_topic][old_emotion] -= 1
                        self.C_LET_total[location][old_emotion] -= 1

                    [z, emotion, c] = self.SampleVZSC(word, location, self.timestamp[i], user)
                    self.ZW[i][j] = z
                    self.E[i][j] = emotion
                    self.Etype[i][j] = c

                    self.C_UC[user][c] += 1
                    self.C_KEW[z][emotion][word] += 1
                    self.C_KEW_total[z][emotion] += 1
                    self.C_DK[i][z] += 1

                    if c == 0:
                        self.C_UE[user][emotion] += 1
                        self.C_UE_total[user] += 1
                        self.C_UET[user][z][emotion] += 1
                        self.C_UET_total[user][emotion] += 1
                    else:
                        self.C_LE[location][emotion] += 1
                        self.C_LE_total[location] += 1
                        self.C_LET[location][z][emotion] += 1
                        self.C_LET_total[location][emotion] += 1

                if len(self.ZW[i]) > 0:
                    self.DZ[i] = Counter(self.ZW[i]).most_common()[0][0]

            if n > self.config['burn_in']:
                self.Psi = self.GetMethodOfMomentsEstimatesForPsi()
                self.Betafunc_Psi = [scipy.special.beta(self.Psi[z][0], self.Psi[z][1]) for z in range(self.K)]

            if n % self.config['sample_lag'] == 0 and n > self.config['burn_in']:
                self.updateParams()

        self.updateParams()
        self.updateDistributions()

    def GetTopicTimestamps(self):
        topic_timestamps = []
        for z in range(self.K):
            l = [i for i in range(len(self.DZ)) if self.DZ[i] == z]
            current_topic_timestamps = self.timestamp[l]
            topic_timestamps.append(current_topic_timestamps)
        return topic_timestamps

    def GetMethodOfMomentsEstimatesForPsi(self):
        topic_timestamps = self.GetTopicTimestamps()
        psi = [[1 for _ in range(2)] for _ in range(len(topic_timestamps))]
        for i in range(len(topic_timestamps)):
            current_topic_timestamps = topic_timestamps[i]
            timestamp_mean = np.mean(current_topic_timestamps)
            timestamp_var = np.var(current_topic_timestamps)
            if timestamp_var == 0:
                timestamp_var = 1e-6
            common_factor = timestamp_mean * (1 - timestamp_mean) / timestamp_var - 1
            psi[i][0] = 1 + timestamp_mean * common_factor
            psi[i][1] = 1 + (1 - timestamp_mean) * common_factor
        return psi

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
        temp = self.C_KEW.sum(axis=1) + self.lambda_TS
        row_sum = temp.sum(axis=1)
        self.phiTSum += (temp / row_sum[:, np.newaxis])

        # Emotion distribution over per topic
        temp = self.C_KEW.sum(axis=2) + np.array(self.beta_E)
        row_sum = temp.sum(axis=1)
        self.phiESum += temp / row_sum[:, np.newaxis]

        # Texual word distribution over emotion
        temp = self.C_KEW.sum(axis=0) + self.config['lambda_TE']
        row_sum = temp.sum(axis=1)
        self.phiTESum += temp / row_sum[:, np.newaxis]

        self.Z_prob = np.add(self.Z_prob, self.C_DK)

        self.updateParamCount += 1

    def SampleVZSC(self, w, location, t, user): # given w and its location list, calculate p(v,z,s,c|w,vlist), sample v,z,s,c
        #C
        first = (self.C_UC[user] + 0.1)

        # K*E
        forth = (self.C_KEW[:, :, w] + self.lambda_TS) / (
                self.C_KEW_total + self.lambda_TS * self.T_size)  # fourth numerator and denominator

        fifth = []
        for z in range(self.K):
            if self.Betafunc_Psi[z] != 0 and t != 0 and t != 1:
                fifth.append((((1 - t) ** (self.Psi[z][0] - 1)) * ((t) ** (self.Psi[z][1] - 1))/self.Betafunc_Psi[z]))  # 5th numerator and denominator
            else:
                fifth.append(1)

        fifth = np.array(fifth)
        newc = np.argmax(np.random.multinomial(1, first/first.sum()))

        if newc == 0:
            # E
            second0 = (self.C_UE[user] + self.beta_E) / (
                    self.C_UE_total[
                        user] + self.beta_E * self.no_of_emotions)  # second numerator and denominator for c = 0
            # K*E
            third0 = (self.C_UET[user] + self.beta_Z) / ((
                                                                 self.C_UET_total[user] + self.beta_Z * self.K)[np.newaxis, :])
            p = forth * third0 * second0[np.newaxis, :] * fifth[:, np.newaxis]
        else:
            #E
            second1 = (self.C_LE[location] + self.beta_E) / (self.C_LE_total[
                                                                location] + self.beta_E * self.no_of_emotions)  # second numerator and denominator for c = 1
            #K*E
            third1 = (self.C_LET[location] + self.beta_Z) / ((
                                                                     self.C_LET_total[location] + self.beta_Z * self.K)[np.newaxis, :])
            p = forth * third1 * second1[np.newaxis, :] * fifth[:, np.newaxis]
        p1 = p.flatten()
        new = np.argmax(np.random.multinomial(1, p1 / p1.sum()))
        return int(new / self.no_of_emotions), new % self.no_of_emotions, newc