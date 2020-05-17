import src.evaluation.util.preprocessing as preprocess_util
import src.evaluation.util.results as result_util
from src.TopicModels.baseE.TS import TS
from src.TopicModels.baseE.LDST import LDST
from src.TopicModels.baseE.JST import JST
from src.TopicModels.baseE.MJST import MJST
from src.TopicModels.baseE.BST import BST
from src.TopicModels.MOSAIC import MOSAIC
import datetime
import src.util.utils as util
import src.evaluation.metrics.coherence as coherence
from src.TopicModels.baseT.TwitterLDA import TwitterLDA
from src.TopicModels.baseT.MRF_LDA import MRF_LDA
from sklearn.decomposition import LatentDirichletAllocation
import src.evaluation.util.utils as eval_utils
import numpy as np

print("Started at: ", datetime.datetime.now())
config = util.readConfig()
MEM, EvaluationData = preprocess_util.readAndProcessData(config)
no_of_iteration = config['evaluation']['no_of_iteration']
evaluation = []

print("%%%%%%%%%%%%%%%%MOSAIC%%%%%%%%%%%%%%%%%%%")
for k in range(2, 11):
    for iteration in range(no_of_iteration):
        print("Iteration: ", iteration)
        count = 0
        for lda_data in MEM:
            print("----------------------------------------")
            print("LDA in interval: ", count)
            print("Begins at: ", datetime.datetime.now())
            lda = MOSAIC(count, k, lda_data, config['gibbs_sampling'])
            lda.inference()
            EvaluationData[count].setK(k)
            EvaluationData[count].setPhiT(lda.phiT)
            EvaluationData[count].setphiTE(lda.phiTE)
            EvaluationData[count].setphiE(lda.phiE)
            EvaluationData[count].setZ(lda.Z)
            EvaluationData[count].setphiTG(lda.phiTG)
            count += 1
            print("Finishes at: ", datetime.datetime.now())

        evl_scoresT = ["MOSAIC"]
        evl_scoresT.append(str(k))
        evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
        evaluation.append(evl_scoresT)
        result_util.saveSensitivityAnalysis(evaluation, config, "topic_Sensitivity")

print("%%%%%%%%%%%%%%%%JST%%%%%%%%%%%%%%%%%%%")
for k in range(2, 11):
    for iteration in range(no_of_iteration):
        print("Iteration: ", iteration)
        count = 0
        print("Started LDA Learning at ", datetime.datetime.now())
        for lda_data in MEM:
            print("----------------------------------------")
            print("LDA in interval: ", count)
            print("Begins at: ", datetime.datetime.now())
            lda = JST(count, k, lda_data , config['gibbs_sampling'], EvaluationData[count].Words)
            lda.inference()
            EvaluationData[count].setK(k)
            EvaluationData[count].setPhiT(lda.phiT)
            EvaluationData[count].setphiTE(lda.phiTE)
            EvaluationData[count].setphiE(lda.phiE)
            count +=1
            print("Finishes at: ", datetime.datetime.now())

        evl_scoresT = ["JST"]
        evl_scoresT.append(str(k))
        evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
        evaluation.append(evl_scoresT)
        result_util.saveSensitivityAnalysis(evaluation, config, "topic_Sensitivity")

print("%%%%%%%%%%%%%%%%TS%%%%%%%%%%%%%%%%%%%")
for k in range(2, 11):
    for iteration in range(no_of_iteration):
        print("Iteration: ", iteration)
        count = 0
        print("Started LDA Learning at ", datetime.datetime.now())
        for lda_data in MEM:
            print("----------------------------------------")
            print("LDA in interval: ", count)
            print("Begins at: ", datetime.datetime.now())
            lda = TS(count, k, lda_data, config['gibbs_sampling'], EvaluationData[count].Words)
            lda.inference()
            EvaluationData[count].setK(k)
            EvaluationData[count].setPhiT(lda.phiT)
            EvaluationData[count].setphiTE(lda.phiTE)
            EvaluationData[count].setphiE(lda.phiE)
            count +=1
            print("Finishes at: ", datetime.datetime.now())

        evl_scoresT = ["TS"]
        evl_scoresT.append(str(k))
        evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
        evaluation.append(evl_scoresT)
        result_util.saveSensitivityAnalysis(evaluation, config, "topic_Sensitivity")

print("%%%%%%%%%%%%%%%%LDST%%%%%%%%%%%%%%%%%%%")
for k in range(2, 11):
    for iteration in range(no_of_iteration):
        print("Iteration: ", iteration)
        count = 0
        print("Started LDA Learning at ", datetime.datetime.now())
        for lda_data in MEM:
            print("----------------------------------------")
            print("LDA in interval: ", count)
            print("Begins at: ", datetime.datetime.now())
            lda = LDST(count, k, lda_data , config['gibbs_sampling'], EvaluationData[count].Words)
            lda.inference()
            EvaluationData[count].setK(k)
            EvaluationData[count].setPhiT(lda.phiT)
            EvaluationData[count].setphiTE(lda.phiTE)
            EvaluationData[count].setphiE(lda.phiE)
            count +=1
            print("Finishes at: ", datetime.datetime.now())

        evl_scoresT = ["LDST"]
        evl_scoresT.append(str(k))
        evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
        evaluation.append(evl_scoresT)
        result_util.saveSensitivityAnalysis(evaluation, config, "topic_Sensitivity")

print("%%%%%%%%%%%%%%%%MJST%%%%%%%%%%%%%%%%%%%")
for k in range(2, 11):
    for iteration in range(no_of_iteration):
        print("Iteration: ", iteration)
        count = 0
        print("Started LDA Learning at ", datetime.datetime.now())
        lda_for_eval = []
        for lda_data in MEM:
            print("----------------------------------------")
            print("LDA in interval: ", count)
            print("Begins at: ", datetime.datetime.now())
            lda = MJST(count, k, lda_data , config['gibbs_sampling'])
            lda.inference()
            EvaluationData[count].setK(k)
            EvaluationData[count].setPhiT(lda.phiT)
            EvaluationData[count].setphiTE(lda.phiTE)
            EvaluationData[count].setphiE(lda.phiE)
            count += 1
            print("Finishes at: ", datetime.datetime.now())

        evl_scoresT = ["MJST"]
        evl_scoresT.append(str(k))
        evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
        evaluation.append(evl_scoresT)
        result_util.saveSensitivityAnalysis(evaluation, config, "topic_Sensitivity")

print("%%%%%%%%%%%%%%%%BST%%%%%%%%%%%%%%%%%%%")
for k in range(2, 11):
    for iteration in range(no_of_iteration):
        print("Iteration: ", iteration)
        count = 0
        for lda_data in MEM:
            print("----------------------------------------")
            print("LDA in interval: ", count)
            print("Begins at: ", datetime.datetime.now())
            lda = BST(count, k, lda_data, config['gibbs_sampling'], EvaluationData[count].Words)
            lda.inference()
            EvaluationData[count].setK(k)
            EvaluationData[count].setPhiT(lda.phiT)
            EvaluationData[count].setphiTE(lda.phiTE)
            EvaluationData[count].setphiE(lda.phiE)
            count +=1
            print("Finishes at: ", datetime.datetime.now())

        evl_scoresT = ["BST"]
        evl_scoresT.append(str(k))
        evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
        evaluation.append(evl_scoresT)
        result_util.saveSensitivityAnalysis(evaluation, config, "topic_Sensitivity")

print("%%%%%%%%%%%%%%%%LDA-Aggregated%%%%%%%%%%%%%%%%%%%")
docs, doc_sizes = eval_utils.getDocsGroupByHashtags(EvaluationData) #del after usage
for k in range(2, 11):
    for iteration in range(no_of_iteration):
        print("Iteration: ", iteration)
        print("Started LDA Learning at ", datetime.datetime.now())
        for count in range(len(EvaluationData)):
            model = LatentDirichletAllocation(n_components=k, learning_method='batch', topic_word_prior=config['gibbs_sampling']['lambda_TS'])
            W = model.fit_transform(docs[count])
            dis = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
            EvaluationData[count].setK(k)
            EvaluationData[count].setPhiT(dis)

        print("Finished LDA Learning at ", datetime.datetime.now())
        evl_scoresT = ["LDA-Aggregated"]
        evl_scoresT.append(str(k))
        evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
        evaluation.append(evl_scoresT)
        result_util.saveSensitivityAnalysis(evaluation, config, "topic_Sensitivity")

print("%%%%%%%%%%%%%%%%ETM%%%%%%%%%%%%%%%%%%%")
docs, edges = eval_utils.getDocsGroupByEmbedding(EvaluationData, doc_sizes) #del after usage
for k in range(2, 11):
    for iteration in range(no_of_iteration):
        print("Iteration: ", iteration)
        count = 0
        print("Started LDA Learning at ", datetime.datetime.now())
        for lda_data in MEM:
            print("----------------------------------------")
            print("LDA in interval: ", count)
            print("Begins at: ", datetime.datetime.now())
            lda = MRF_LDA(count, k, lda_data , config['gibbs_sampling'], docs[count], edges[count])
            lda.inference()
            EvaluationData[count].setK(k)
            EvaluationData[count].setPhiT(lda.phiT)
            EvaluationData[count].setZ(lda.Z)
            count +=1
            print("Finishes at: ", datetime.datetime.now())

        print("Finished LDA Learning at ", datetime.datetime.now())
        evl_scoresT = ["ETM"]
        evl_scoresT.append(str(k))
        evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
        evaluation.append(evl_scoresT)
        result_util.saveSensitivityAnalysis(evaluation, config, "topic_Sensitivity")

del docs
del edges

print("%%%%%%%%%%%%%%%%Twitter-LDA%%%%%%%%%%%%%%%%%%%")
for k in range(2, 11):
    for iteration in range(no_of_iteration):
        print("Iteration: ", iteration)
        count = 0
        for lda_data in MEM:
            print("----------------------------------------")
            print("LDA in interval: ", count)
            print("Begins at: ", datetime.datetime.now())
            lda = TwitterLDA(count, k, lda_data , config['gibbs_sampling'])
            lda.inference()
            EvaluationData[count].setK(k)
            EvaluationData[count].setPhiT(lda.phiT)
            EvaluationData[count].setZ(lda.Z)
            count +=1
            print("Finishes at: ", datetime.datetime.now())

        print("Finished LDA Learning at ", datetime.datetime.now())
        evl_scoresT = ["Twitter-LDA"]
        evl_scoresT.append(str(k))
        evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
        evaluation.append(evl_scoresT)
        result_util.saveSensitivityAnalysis(evaluation, config, "topic_Sensitivity")
