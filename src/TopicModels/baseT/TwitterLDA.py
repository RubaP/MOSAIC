import numpy as np
from random import choices

class TwitterLDA:
    def __init__(self, id, K, lda_data, config):
        print("-----STARTED LDA INITIALIZATION-----")
        print("No of users: ", lda_data['no_of_users'])

        data = lda_data["data"].reset_index().copy()

        self.id = id
        self.interval = id
        self.K = K
        self.users = lda_data['users']
        self.U_size = lda_data['no_of_users']
        self.textual_vocabulary = lda_data["textual_vocabulary"]
        self.config = config
        self.T_size = len(lda_data["textual_vocabulary"])
        self.no_of_doc = data.shape[0]
        print("Number of topic: ", K)

        #Textual words
        self.Words = []

        #Four assignment variables of the model
        self.T = []
        self.Z = []

        #Stabilization variables
        self.phiTSum = np.zeros((self.K, self.T_size))
        self.updateParamCount = 0

        #Count matrices
        self.C_WT_D = np.zeros(2)
        self.C_WS = np.zeros((K, self.T_size))

        self.C_WG = np.zeros(self.T_size)
        self.C_DZ = np.zeros((self.U_size, K))

        #Total value matrices
        self.C_WG_total = 0
        self.C_WS_total = np.zeros(K)

        self.Z_prob = np.zeros((self.no_of_doc, self.K))

        for i, row in data.iterrows():
            user = self.users[i]
            # Do topic assignment
            topic = np.random.randint(K)
            self.Z.append(topic)
            self.C_DZ[user][topic] += 1

            T_assignment_list = []
            words_list = []

            # Do word type assignment
            text = row['cleanedText']
            for t_word in text:
                if t_word in self.textual_vocabulary:
                    t = self.textual_vocabulary.index(t_word)
                    T_assignment = choices(range(2), np.random.dirichlet(self.config['beta_T'][0:2], 1)[0, :])[0]

                    self.C_WT_D[T_assignment] += 1
                    if T_assignment == 0:
                        self.C_WS[topic][t] += 1
                        self.C_WS_total[topic] += 1
                    elif T_assignment == 1:
                        self.C_WG[t] += 1
                        self.C_WG_total += 1

                    T_assignment_list.append(T_assignment)
                    words_list.append(t)

            hashtags = row['hashtags']
            for t_word in hashtags:
                if t_word in self.textual_vocabulary:
                    t = self.textual_vocabulary.index(t_word)
                    T_assignment = choices(range(2), np.random.dirichlet(self.config['beta_T'][0:2], 1)[0, :])[0]

                    self.C_WT_D[T_assignment] += 1
                    if T_assignment == 0:
                        self.C_WS[topic][t] += 1
                        self.C_WS_total[topic] += 1
                    elif T_assignment == 1:
                        self.C_WG[t] += 1
                        self.C_WG_total += 1

                    words_list.append(t)
                    T_assignment_list.append(T_assignment)

            emojis = row['emojis']
            for t_word in emojis:
                if t_word in self.textual_vocabulary:
                    t = self.textual_vocabulary.index(t_word)
                    T_assignment = choices(range(2), np.random.dirichlet(self.config['beta_T'][0:2], 1)[0, :])[0]

                    self.C_WT_D[T_assignment] += 1
                    if T_assignment == 0:
                        self.C_WS[topic][t] += 1
                        self.C_WS_total[topic] += 1
                    elif T_assignment == 1:
                        self.C_WG[t] += 1
                        self.C_WG_total += 1

                    words_list.append(t)
                    T_assignment_list.append(T_assignment)

            self.T.append(T_assignment_list)
            self.Words.append(words_list)

        print("WS Total: ",self.C_WS_total)
        print("WG Total: ",self.C_WG_total)
        print("======================")
        print("+++++INITIALIZATION COMPLETED+++++")

    def inference(self):
        iterations = self.config["maximum_iteration"]

        print("++++++START SAMPLING+++++++++")
        print("No of documents: ", self.no_of_doc)
        for n in range(iterations):
            self.previous_Z = self.Z.copy()
            for i in range(self.no_of_doc):
                user = self.users[i]
                # remove current doc from assignment Z
                current_topic = self.Z[i]
                self.C_DZ[user][current_topic] -= 1
                T_assignment = self.T[i]
                no_of_Tword = len(T_assignment)

                for j in range(no_of_Tword):
                    word = self.Words[i][j]
                    self.removeTopicRelatedAssignments(j, word, T_assignment, current_topic)

                #FINE NEW TOPIC
                new_topic = self.find_topic(self.Words[i], T_assignment, user)
                self.updateNewTopic(i, new_topic, user)

                #UPDATE TEXTUAL WORD ASSIGNMENT
                for j in range(no_of_Tword):
                    word = self.Words[i][j]
                    self.removeWordRelatedAssignment(i, j, word, T_assignment)
                    new_t = self.find_t_assignment(i, new_topic, word, j)
                    self.update_t(new_t, i, j, new_topic, word)

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

        self.updateParamCount += 1

    def updateNewTopic(self, i, topic, user):
        self.Z[i] = topic
        self.C_DZ[user][topic] += 1

    def removeWordRelatedAssignment(self, i, j, word, T_assignment):
        self.C_WT_D[T_assignment[j]] -= 1
        # WS_total and WE_total are already substracted
        if T_assignment[j] == 1:
            self.C_WG[word] -= 1
            self.C_WG_total -= 1

    def removeTopicRelatedAssignments(self, j, word, T_assignment, topic):
        if T_assignment[j] == 0:
            self.C_WS[topic][word] -= 1
            self.C_WS_total[topic] -= 1

    def find_topic(self, W, T_assignment, user):
        p_z = (self.config['beta_Z'] + self.C_DZ[user])
        for j in range(len(T_assignment)):
            if T_assignment[j] == 0:
                p_z = p_z*(self.config['lambda_TS'] + self.C_WS[:, W[j]])/(self.config['lambda_TS']* self.T_size + self.C_WS_total)

        new_z = np.argmax(np.random.multinomial(1, p_z / p_z.sum()))
        return new_z

    def find_t_assignment(self, i, topic, word, j):
        p_S = (self.config['beta_T'][0] + self.C_WT_D[0])*((self.config['lambda_TS'] + self.C_WS[topic][word])/(self.config['lambda_TS']*self.T_size + self.C_WS_total[topic]))
        p_G = (self.config['beta_T'][1] + self.C_WT_D[1])*((self.config['lambda_TG'] + self.C_WG[word])/(self.config['lambda_TG']* self.T_size + self.C_WG_total))

        p = [p_S, p_G]/(p_S + p_G)
        return np.argmax(np.random.multinomial(1, p))

    def update_t(self, t, i, j, topic, word):
        self.T[i][j] = t
        self.C_WT_D[t] += 1

        if t == 0:
            self.C_WS[topic][word] += 1
            self.C_WS_total[topic] += 1
        if t == 1:
            self.C_WG[word] += 1
            self.C_WG_total += 1

    def updateDistributions(self):
        self.Z = np.argmax(self.Z_prob, axis=1)
        self.phiT = self.phiTSum / self.updateParamCount