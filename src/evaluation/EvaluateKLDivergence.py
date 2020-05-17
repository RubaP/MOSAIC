import src.evaluation.util.preprocessing as preprocess_util
import src.evaluation.util.results as result_util
from src.TopicModels.baseE.TS import TS
from src.TopicModels.baseE.LDST import LDST
from src.TopicModels.baseE.JST import JST
from src.TopicModels.baseE.MJST import MJST
from src.TopicModels.baseE.BST import BST
import datetime
import src.util.utils as util
import src.evaluation.metrics.coherence as coherence
import src.evaluation.metrics.KLDivergence as emDis
from src.TopicModels.baseT.TwitterLDA import TwitterLDA
from src.TopicModels.baseT.MRF_LDA import MRF_LDA
import src.evaluation.util.utils as eval_utils
from sklearn.decomposition import LatentDirichletAllocation
from src.TopicModels.MOSAIC import MOSAIC
import numpy as np

print("Started at: ", datetime.datetime.now())
config = util.readConfig()

MEM, EvaluationData = preprocess_util.readAndProcessData(config)
no_of_iteration = config['evaluation']['no_of_iteration']

evaluation = []
evaluationEmDis = []

print("%%%%%%%%%%%%%%%%Twitter-LDA%%%%%%%%%%%%%%%%%%%")
for iteration in range(no_of_iteration):
    print("Iteration: ", iteration)
    count = 0
    for lda_data in MEM:
        print("----------------------------------------")
        print("LDA in interval: ", count)
        print("Begins at: ", datetime.datetime.now())
        lda = TwitterLDA(count, config['topics'], lda_data , config['gibbs_sampling'])
        lda.inference()
        EvaluationData[count].setPhiT(lda.phiT)
        EvaluationData[count].setZ(lda.Z)
        count +=1
        print("Finishes at: ", datetime.datetime.now())

    print("Finished LDA Learning at ", datetime.datetime.now())
    evl_scores = ["Twitter-LDA"]
    evl_scores.extend(coherence.compute_coherence(EvaluationData))
    evaluation.append(evl_scores)
    result_util.saveEvaluations(evaluation, config, "topic_baseT")

print("%%%%%%%%%%%%%%%%LDA-Aggregated%%%%%%%%%%%%%%%%%%%")
docs, doc_sizes = eval_utils.getDocsGroupByHashtags(EvaluationData) #del after usage
for iteration in range(no_of_iteration):
    print("Iteration: ", iteration)
    print("Started LDA Learning at ", datetime.datetime.now())
    for count in range(len(EvaluationData)):
        model = LatentDirichletAllocation(n_components=config['topics'], learning_method='batch', topic_word_prior=config['gibbs_sampling']['lambda_TS'])
        W = model.fit_transform(docs[count])
        dis = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
        EvaluationData[count].setPhiT(dis)

    print("Finished LDA Learning at ", datetime.datetime.now())
    evl_scores = ["LDA-Aggregated"]
    evl_scores.extend(coherence.compute_coherence(EvaluationData))
    evaluation.append(evl_scores)
    result_util.saveEvaluations(evaluation, config, "topic_baseT")

print("%%%%%%%%%%%%%%%%ETM%%%%%%%%%%%%%%%%%%%")
docs, edges = eval_utils.getDocsGroupByEmbedding(EvaluationData, doc_sizes) #del after usage
for iteration in range(no_of_iteration):
    print("Iteration: ", iteration)
    count = 0
    print("Started LDA Learning at ", datetime.datetime.now())
    for lda_data in MEM:
        print("----------------------------------------")
        print("LDA in interval: ", count)
        print("Begins at: ", datetime.datetime.now())
        lda = MRF_LDA(count, config['topics'], lda_data , config['gibbs_sampling'], docs[count], edges[count])
        lda.inference()
        EvaluationData[count].setPhiT(lda.phiT)
        EvaluationData[count].setZ(lda.Z)
        count +=1
        print("Finishes at: ", datetime.datetime.now())

    print("Finished LDA Learning at ", datetime.datetime.now())
    evl_scores = ["ETM"]
    evl_scores.extend(coherence.compute_coherence(EvaluationData))
    evaluation.append(evl_scores)
    result_util.saveEvaluations(evaluation, config, "topic_baseT")
del docs
del edges

print("%%%%%%%%%%%%%%%%JST%%%%%%%%%%%%%%%%%%%")
for iteration in range(no_of_iteration):
    print("Iteration: ", iteration)
    count = 0
    print("Started LDA Learning at ", datetime.datetime.now())
    for lda_data in MEM:
        print("----------------------------------------")
        print("LDA in interval: ", count)
        print("Begins at: ", datetime.datetime.now())
        lda = JST(count, config['topics'], lda_data , config['gibbs_sampling'], EvaluationData[count].Words)
        lda.inference()
        EvaluationData[count].setPhiT(lda.phiT)
        EvaluationData[count].setphiTE(lda.phiTE)
        EvaluationData[count].setphiE(lda.phiE)
        count +=1
        print("Finishes at: ", datetime.datetime.now())

    evl_scoreDis = ["JST"]
    evl_scoreDis.append(emDis.compute_divergence(EvaluationData))
    evaluationEmDis.append(evl_scoreDis)
    result_util.saveEmDisDivergence(evaluationEmDis, config, "emDistribution_baseE")

    evl_scoresT = ["JST"]
    evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
    evaluation.append(evl_scoresT)
    result_util.saveEvaluations(evaluation, config, "topic_baseE")

print("%%%%%%%%%%%%%%%%TS%%%%%%%%%%%%%%%%%%%")
for iteration in range(no_of_iteration):
    print("Iteration: ", iteration)
    count = 0
    print("Started LDA Learning at ", datetime.datetime.now())
    for lda_data in MEM:
        print("----------------------------------------")
        print("LDA in interval: ", count)
        print("Begins at: ", datetime.datetime.now())
        lda = TS(count, config['topics'], lda_data, config['gibbs_sampling'], EvaluationData[count].Words)
        lda.inference()
        EvaluationData[count].setPhiT(lda.phiT)
        EvaluationData[count].setphiTE(lda.phiTE)
        EvaluationData[count].setphiE(lda.phiE)
        count +=1
        print("Finishes at: ", datetime.datetime.now())

    evl_scoreDis = ["TS"]
    evl_scoreDis.append(emDis.compute_divergence(EvaluationData))
    evaluationEmDis.append(evl_scoreDis)
    result_util.saveEmDisDivergence(evaluationEmDis, config, "emDistribution_baseE")

    evl_scoresT = ["TS"]
    evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
    evaluation.append(evl_scoresT)
    result_util.saveEvaluations(evaluation, config, "topic_baseE")

print("%%%%%%%%%%%%%%%%BST%%%%%%%%%%%%%%%%%%%")
for iteration in range(no_of_iteration):
    print("Iteration: ", iteration)
    count = 0
    for lda_data in MEM:
        print("----------------------------------------")
        print("LDA in interval: ", count)
        print("Begins at: ", datetime.datetime.now())
        lda = BST(count, config['topics'], lda_data, config['gibbs_sampling'], EvaluationData[count].Words)
        lda.inference()
        EvaluationData[count].setPhiT(lda.phiT)
        EvaluationData[count].setphiTE(lda.phiTE)
        EvaluationData[count].setphiE(lda.phiE)
        count +=1
        print("Finishes at: ", datetime.datetime.now())

    evl_scoreDis = ["BST"]
    evl_scoreDis.append(emDis.compute_divergence(EvaluationData))
    evaluationEmDis.append(evl_scoreDis)
    result_util.saveEmDisDivergence(evaluationEmDis, config, "emDistribution_baseE")

    evl_scoresT = ["BST"]
    evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
    evaluation.append(evl_scoresT)
    result_util.saveEvaluations(evaluation, config, "topic_baseE")

print("%%%%%%%%%%%%%%%%LDST%%%%%%%%%%%%%%%%%%%")
for iteration in range(no_of_iteration):
    print("Iteration: ", iteration)
    count = 0
    print("Started LDA Learning at ", datetime.datetime.now())
    for lda_data in MEM:
        print("----------------------------------------")
        print("LDA in interval: ", count)
        print("Begins at: ", datetime.datetime.now())
        lda = LDST(count, config['topics'], lda_data , config['gibbs_sampling'], EvaluationData[count].Words)
        lda.inference()
        EvaluationData[count].setPhiT(lda.phiT)
        EvaluationData[count].setphiTE(lda.phiTE)
        EvaluationData[count].setphiE(lda.phiE)
        count +=1
        print("Finishes at: ", datetime.datetime.now())

    evl_scoreDis = ["LDST"]
    evl_scoreDis.append(emDis.compute_divergence(EvaluationData))
    evaluationEmDis.append(evl_scoreDis)
    result_util.saveEmDisDivergence(evaluationEmDis, config, "emDistribution_baseE")

    evl_scoresT = ["LDST"]
    evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
    evaluation.append(evl_scoresT)
    result_util.saveEvaluations(evaluation, config, "topic_baseE")

print("%%%%%%%%%%%%%%%%MJST%%%%%%%%%%%%%%%%%%%")
for iteration in range(no_of_iteration):
    print("Iteration: ", iteration)
    count = 0
    print("Started LDA Learning at ", datetime.datetime.now())
    lda_for_eval = []
    for lda_data in MEM:
        print("----------------------------------------")
        print("LDA in interval: ", count)
        print("Begins at: ", datetime.datetime.now())
        lda = MJST(count, config['topics'], lda_data , config['gibbs_sampling'])
        lda.inference()
        EvaluationData[count].setPhiT(lda.phiT)
        EvaluationData[count].setphiTE(lda.phiTE)
        EvaluationData[count].setphiE(lda.phiE)
        count += 1
        print("Finishes at: ", datetime.datetime.now())

    evl_scoreDis = ["MJST"]
    evl_scoreDis.append(emDis.compute_divergence(EvaluationData))
    evaluationEmDis.append(evl_scoreDis)
    result_util.saveEmDisDivergence(evaluationEmDis, config, "emDistribution_baseE")

    evl_scoresT = ["MJST"]
    evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
    evaluation.append(evl_scoresT)
    result_util.saveEvaluations(evaluation, config, "topic_baseE")

print("%%%%%%%%%%%%%%%%MOSAIC%%%%%%%%%%%%%%%%%%%")
for iteration in range(no_of_iteration):
    print("Iteration: ", iteration)
    count = 0
    for lda_data in MEM:
        print("----------------------------------------")
        print("LDA in interval: ", count)
        print("Begins at: ", datetime.datetime.now())
        lda = MOSAIC(count, config['topics'], lda_data, config['gibbs_sampling'])
        lda.inference()
        EvaluationData[count].setPhiT(lda.phiT)
        EvaluationData[count].setphiTE(lda.phiTE)
        EvaluationData[count].setphiE(lda.phiE)
        EvaluationData[count].setZ(lda.Z)
        EvaluationData[count].setphiTG(lda.phiTG)
        count += 1
        print("Finishes at: ", datetime.datetime.now())

    evl_scoresT = ["MOSAIC"]
    evl_scoresT.extend(coherence.compute_coherence(EvaluationData))
    evaluation.append(evl_scoresT)
    result_util.saveEvaluations(evaluation, config, "topic_MOSAIC")

    evl_scoreDis = ["MOSAIC"]
    evl_scoreDis.append(emDis.compute_divergence(EvaluationData))
    evaluationEmDis.append(evl_scoreDis)

    result_util.saveEmDisDivergence(evaluationEmDis, config, "emDistribution_MOSAIC")