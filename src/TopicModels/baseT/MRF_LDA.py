import numpy as np

class MRF_LDA:
    def __init__(self, id, K, lda_data, config, Words, edges):
        print("-----STARTED LDA INITIALIZATION-----")

        self.id = id
        self.interval = id
        self.K = K
        self.textual_vocabulary = lda_data["textual_vocabulary"]
        self.config = config
        self.T_size = len(lda_data["textual_vocabulary"])
        self.no_of_doc = len(Words)
        self.edges = edges
        print("Number of topic: ", K)

        #Textual words
        self.Words = Words
        self.ZW = [] #Word level Z assignment

        #Count matrices
        self.C_WS = np.zeros((K, self.T_size))
        self.C_DS = np.zeros((self.no_of_doc, K))

        #Total value matrices
        self.C_WS_total = np.zeros(K)

        # Stabilization variables
        self.phiTSum = np.zeros((self.K, self.T_size))
        self.updateParamCount = 0
        self.Z_prob = np.zeros((self.no_of_doc, self.K))

        for i in range(self.no_of_doc):
            words_list = Words[i]
            z_list = []

            for t in words_list:
                topic = np.random.randint(K)
                self.C_WS[topic][t] += 1
                self.C_DS[i][topic] += 1
                self.C_WS_total[topic] += 1

                z_list.append(topic)
            self.ZW.append(z_list)

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

                    self.C_WS[old_topic][word] -= 1
                    self.C_DS[i][old_topic] -= 1
                    self.C_WS_total[old_topic] -= 1

                    topic = self.find_topic(i, word, self.ZW[i], self.edges[i][j])
                    self.C_WS[topic][word] += 1
                    self.C_DS[i][topic] += 1
                    self.C_WS_total[topic] += 1
                    self.ZW[i][j] = topic
            if n % self.config['sample_lag'] == 0 and n > self.config['burn_in']:
                self.updateParams()

        self.updateParams()
        self.updateDistributions()

    def find_topic(self, i, word, Z, edges):
        lamda = 1 #if lamda is zero then the prob will be same as LDA

        prob = np.array([1]*self.K)
        count = len(edges)
        if count > 0:
            for edge in edges:
                prob[Z[edge]] +=1
            prob = np.exp(lamda*prob/count)

        p_z = (self.config['beta_Z'] + self.C_DS[i])*(self.config['lambda_TS'] + self.C_WS[:, word])/(self.config['lambda_TS']* self.T_size + self.C_WS_total)
        p_z = np.multiply(p_z, prob)
        new_z = np.argmax(np.random.multinomial(1, p_z / p_z.sum()))
        return new_z

    def updateDistributions(self):
        self.Z = np.argmax(self.Z_prob, axis=1)
        self.phiT = self.phiTSum / self.updateParamCount

    def updateParams(self):
        self.Z_prob = np.add(self.Z_prob, self.C_DS)
        temp = self.C_WS + self.config['lambda_TS']
        row_sum = temp.sum(axis=1)
        self.phiTSum += (temp / row_sum[:, np.newaxis])

        self.updateParamCount += 1