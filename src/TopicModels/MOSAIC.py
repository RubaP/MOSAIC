import numpy as np
import random
from random import choices

class MOSAIC:
    def __init__(self, id, K, lda_data, config):
        print("-----STARTED LDA INITIALIZATION-----")
        print("configuration: ", config)
        data = lda_data["data"].reset_index().copy()
        emotionalSeedWords = lda_data["emotionalSeedWords"]
        self.genericSeedWords = lda_data["genericSeedWords"]

        self.id = id
        self.interval = id
        self.K = K
        self.textual_vocabulary = lda_data["textual_vocabulary"]
        self.config = config
        self.T_size = len(lda_data["textual_vocabulary"])
        self.no_of_doc = data.shape[0]
        self.no_of_emotions = 6
        print("Number of topic: ", K)

        #Textual words
        self.Words = []

        #Four assignment variables of the model
        self.T = []
        self.Z = []
        self.E = []

        #Count matrices
        self.C_WT_D = np.zeros((self.no_of_doc, 3))
        self.C_WS = np.zeros((K, self.T_size))
        self.C_WG = np.zeros(self.T_size)
        self.C_WE = np.zeros((self.no_of_emotions, self.T_size))
        self.C_ZE = np.zeros((K, self.no_of_emotions))
        self.C_DZ = np.zeros(K)

        #Total value matrices
        self.C_WG_total = 0
        self.C_WS_total = np.zeros(K)
        self.C_WE_total = np.zeros(self.no_of_emotions)
        self.C_ZE_total = np.zeros(K)

        self.Z_prob = np.zeros((self.no_of_doc, self.K))
        self.phiTSum = np.zeros((self.K, self.T_size))
        self.phiTGSum = np.zeros(self.T_size)
        self.phiTESum = np.zeros((self.no_of_emotions, self.T_size))
        self.phiESum = np.zeros((K, self.no_of_emotions))

        self.updateParamCount = 0

        for i, row in data.iterrows():
            topic = np.random.randint(K)
            self.Z.append(topic)
            self.C_DZ[topic] += 1

            T_assignment_list = []
            E_assignment_list = []
            words_list = []

            # Do word type assignment
            text = row['cleanedText']
            for t_word in text:
                if t_word in self.textual_vocabulary:
                    t = self.textual_vocabulary.index(t_word)
                    emotion = -1
                    if t_word in self.genericSeedWords:
                        T_assignment = 1
                    elif t_word in emotionalSeedWords.keys():
                        T_assignment = 2
                        emotion = random.choice(emotionalSeedWords[t_word])
                    else:
                        T_assignment = choices(range(3), np.random.dirichlet(self.config['beta_T'], 1)[0, :])[0]

                    self.C_WT_D[i][T_assignment] += 1
                    if T_assignment == 0:
                        self.C_WS[topic][t] += 1
                        self.C_WS_total[topic] += 1
                    elif T_assignment == 1:
                        self.C_WG[t] += 1
                        self.C_WG_total += 1
                    else:
                        if emotion == -1:
                            emotion = choices(range(6), weights = self.C_WE_total + self.config['beta_TE'])[0]
                        self.C_WE[emotion][t] += 1
                        self.C_WE_total[emotion] += 1
                        self.C_ZE[topic][emotion] += 1
                        self.C_ZE_total[topic] += 1

                    T_assignment_list.append(T_assignment)
                    words_list.append(t)
                    E_assignment_list.append(emotion)

            hashtags = row['hashtags']
            for t_word in hashtags:
                if t_word in self.textual_vocabulary:
                    t = self.textual_vocabulary.index(t_word)
                    emotion = -1
                    if t_word in self.genericSeedWords:
                        T_assignment = 1
                    elif t_word in emotionalSeedWords.keys():
                        T_assignment = 2
                        emotion = random.choice(emotionalSeedWords[t_word])
                    else:
                        T_assignment = choices(range(3), np.random.dirichlet(self.config['beta_T'], 1)[0, :])[0]

                    self.C_WT_D[i][T_assignment] += 1
                    if T_assignment == 0:
                        self.C_WS[topic][t] += 1
                        self.C_WS_total[topic] += 1
                    elif T_assignment == 1:
                        self.C_WG[t] += 1
                        self.C_WG_total += 1
                    else:
                        if emotion == -1:
                            emotion = choices(range(6), weights=self.C_WE_total + self.config['beta_TE'])[0]
                        self.C_WE[emotion][t] += 1
                        self.C_WE_total[emotion] += 1
                        self.C_ZE[topic][emotion] += 1
                        self.C_ZE_total[topic] += 1

                    words_list.append(t)
                    T_assignment_list.append(T_assignment)
                    E_assignment_list.append(emotion)

            emojis = row['emojis']
            for t_word in emojis:
                if t_word in self.textual_vocabulary:
                    t = self.textual_vocabulary.index(t_word)
                    emotion = -1
                    if t_word in lda_data['emojiSeedWordsAll'].keys():
                        emotion_prob = lda_data['emojiSeedWordsAll'][t_word]
                        emotion = choices(range(7), weights=list(map(float, emotion_prob)))[0]
                        if emotion == 6:
                            emotion = choices(range(6), weights=self.C_WE_total + self.config['beta_TE'])[0]
                            T_assignment = choices(range(3), np.random.dirichlet(self.config['beta_T'], 1)[0, :])[0]
                        else:
                            T_assignment = 2
                    else:
                        T_assignment = choices(range(3), np.random.dirichlet(self.config['beta_T'], 1)[0, :])[0]


                    self.C_WT_D[i][T_assignment] += 1
                    if T_assignment == 0:
                        self.C_WS[topic][t] += 1
                        self.C_WS_total[topic] += 1
                    elif T_assignment == 1:
                        self.C_WG[t] += 1
                        self.C_WG_total += 1
                    else:
                        if emotion == -1:
                            emotion = choices(range(6), weights=self.C_WE_total + self.config['beta_TE'])[0]
                        self.C_WE[emotion][t] +=1
                        self.C_WE_total[emotion] +=1
                        self.C_ZE[topic][emotion] +=1
                        self.C_ZE_total[topic] +=1

                    words_list.append(t)
                    T_assignment_list.append(T_assignment)
                    E_assignment_list.append(emotion)

            self.T.append(T_assignment_list)
            self.Words.append(words_list)
            self.E.append(E_assignment_list)

        print("WS Total: ",self.C_WS_total)
        print("WG Total: ",self.C_WG_total)
        print("WE Total: ", self.C_WE_total)
        print("======================")
        print("+++++INITIALIZATION COMPLETED+++++")

    def inference(self):
        iterations = self.config["maximum_iteration"]

        print("++++++START SAMPLING+++++++++")
        print("No of documents: ", self.no_of_doc)

        for n in range(iterations):
            for i in range(self.no_of_doc):
                # remove current doc from assignment Z
                current_topic = self.Z[i]
                self.C_DZ[current_topic] -= 1
                T_assignment = self.T[i]
                E_assignment = self.E[i]
                no_of_Tword = len(T_assignment)

                for j in range(no_of_Tword):
                    word = self.Words[i][j]
                    self.removeTopicRelatedAssignments(j, word, T_assignment, E_assignment, current_topic)

                #FINE NEW TOPIC
                new_topic = self.find_topic(self.Words[i], T_assignment, E_assignment)
                self.updateNewTopic(i, new_topic)

                #UPDATE TEXTUAL WORD ASSIGNMENT
                for j in range(no_of_Tword):
                    word = self.Words[i][j]
                    self.removeWordRelatedAssignment(i, j, word, T_assignment, E_assignment)
                    new_emotion = self.find_E_assingment(new_topic, word)
                    new_t = self.find_t_assignment(i, new_topic, word, new_emotion, j)
                    self.update_t(new_t, i, j, new_topic, word, new_emotion)

            if n % self.config['sample_lag'] == 0 and n > self.config['burn_in']:
                self.updateParams()

        self.updateParams()
        self.updateDistributions()

    def updateParams(self):
        for m in range(self.no_of_doc):
            self.Z_prob[m][self.Z[m]] +=1

        # Texual word distribution per topic
        temp = self.C_WS + self.config['lambda_TS']
        row_sum = temp.sum(axis=1)
        self.phiTSum += (temp / row_sum[:, np.newaxis])

        # General word distribution per topic
        temp = self.C_WG + self.config['lambda_TG']
        row_sum = temp.sum()
        self.phiTGSum += temp / row_sum

        # Emotion distribution over per topic
        temp = self.C_ZE + np.array(self.config['beta_E'])
        row_sum = temp.sum(axis=1)
        self.phiESum += temp / row_sum[:, np.newaxis]

        # Texual word distribution over emotion
        temp = self.C_WE + self.config['lambda_TE']
        row_sum = temp.sum(axis=1)
        self.phiTESum += temp / row_sum[:, np.newaxis]

        self.updateParamCount += 1

    def updateNewTopic(self, i, topic):
        self.Z[i] = topic
        self.C_DZ[topic] += 1

    def removeWordRelatedAssignment(self, i, j, word, T_assignment, E_assignment,):
        self.C_WT_D[i][T_assignment[j]] -= 1
        # WS_total and WE_total are already substracted
        if T_assignment[j] == 1:
            self.C_WG[word] -= 1
            self.C_WG_total -= 1
        elif T_assignment[j] == 2:
            emotion = E_assignment[j]
            self.C_WE[emotion][word] -= 1
            self.C_WE_total[emotion] -= 1

    def removeTopicRelatedAssignments(self, j, word, T_assignment, E_assignment, topic):
        if T_assignment[j] == 0:
            self.C_WS[topic][word] -= 1
            self.C_WS_total[topic] -= 1
        elif T_assignment[j] == 2:
            emotion = E_assignment[j]
            self.C_ZE[topic][emotion] -= 1
            self.C_ZE_total[topic] -= 1

    def find_topic(self, W, T_assignment, E_assignment):
        p_z = (self.config['beta_Z'] + self.C_DZ)
        for j in range(len(T_assignment)):
            if T_assignment[j] == 0:
                p_z = p_z*(self.config['lambda_TS'] + self.C_WS[:, W[j]])/(self.config['lambda_TS']* self.T_size + self.C_WS_total)
            elif T_assignment[j] == 2:
                e = E_assignment[j]
                p_z = p_z * (self.config['beta_E'][e] + self.C_ZE[:, e]) / (
                            self.config['beta_E'][e] * self.no_of_emotions + self.C_ZE_total)
        new_z = np.argmax(np.random.multinomial(1, p_z / p_z.sum()))
        return new_z

    def find_t_assignment(self, i, topic, word, emotion, j):
        p_S = (self.config['beta_T'][0] + self.C_WT_D[i][0])*((self.config['lambda_TS'] + self.C_WS[topic][word])/(self.config['lambda_TS']*self.T_size + self.C_WS_total[topic]))
        p_G = (self.config['beta_T'][1] + self.C_WT_D[i][1])*((self.config['lambda_TG'] + self.C_WG[word])/(self.config['lambda_TG']*self.T_size + self.C_WG_total))
        p_E = (self.config['beta_T'][2] + self.C_WT_D[i][2]) *((self.config['lambda_TE'] + self.C_WE[emotion][word])/(self.config['lambda_TE'] *self.T_size + self.C_WE_total[emotion]))

        p = [p_S, p_G, p_E]/(p_S + p_G + p_E)
        return np.argmax(np.random.multinomial(1, p))

    def find_E_assingment(self, new_topic, word):
        p_e = (self.config['beta_E'] + self.C_ZE[new_topic,:])*(self.config['lambda_TE'] + self.C_WE[:, word])/ (
                self.config['lambda_TE']*self.T_size + self.C_WE_total)
        p = p_e / p_e.sum()
        return np.argmax(np.random.multinomial(1, p))

    def update_t(self, t, i, j, topic, word, emotion):
        self.T[i][j] = t
        self.C_WT_D[i][t] += 1
        self.E[i][j] = -1

        if t == 0:
            self.C_WS[topic][word] += 1
            self.C_WS_total[topic] += 1
        if t == 1:
            self.C_WG[word] += 1
            self.C_WG_total += 1
        if t ==2:
            self.E[i][j] = emotion
            self.C_WE[emotion][word] +=1
            self.C_WE_total[emotion] +=1
            self.C_ZE[topic][emotion] +=1
            self.C_ZE_total[topic] +=1

    def updateDistributions(self):
        self.Z = np.argmax(self.Z_prob, axis=1)
        self.phiT = self.phiTSum / self.updateParamCount
        self.phiTG = self.phiTGSum / self.updateParamCount
        self.phiTE = self.phiTESum / self.updateParamCount
        self.phiE = self.phiESum / self.updateParamCount