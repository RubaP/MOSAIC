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
import json
import src.components.contentSelection as content_selection
import src.evaluation.metrics.rougeScore as rougeScore
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

with open(config['path'] + "/groundTruthSummary1.json") as f:
    groundTruthSummary1 = json.load(f)
with open(config['path'] + "/groundTruthSummary2.json") as f:
    groundTruthSummary2 = json.load(f)
with open(config['path'] + "/groundTruthSummary3.json") as f:
    groundTruthSummary3 = json.load(f)
print("Ground truth summary loaded")
fileName = "RougeAnalysisAll"

groundTruthSummary = [groundTruthSummary1, groundTruthSummary2, groundTruthSummary3]

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
    summary = content_selection.generateSummary(EvaluationData)
    evl_scores.extend(
        rougeScore.computeRougeScore(summary, groundTruthSummary, config['topics'], config['intervals']))
    evaluation.append(evl_scores)
    result_util.saveRougeAnalysis(evaluation, config, fileName)

print("%%%%%%%%%%%%%%%%LDA-Aggregated%%%%%%%%%%%%%%%%%%%")
docs, doc_sizes = eval_utils.getDocsGroupByHashtags(EvaluationData) #del after usage
for iteration in range(no_of_iteration):
    print("Iteration: ", iteration)
    print("Started LDA Learning at ", datetime.datetime.now())
    for count in range(len(EvaluationData)):
        data = EvaluationData[count]
        model = LatentDirichletAllocation(n_components=config['topics'], learning_method='batch', topic_word_prior=config['gibbs_sampling']['lambda_TS'])
        W = model.fit_transform(docs[count])
        dis = model.components_ / model.components_.sum(axis=1)[:, np.newaxis]
        T_dis_per_doc = model.transform(data.getT_bow())
        EvaluationData[count].setPhiT(dis)
        EvaluationData[count].setphiE([[None]]*6)
        EvaluationData[count].setZ([np.argmax(x) for x in T_dis_per_doc])

    print("Finished LDA Learning at ", datetime.datetime.now())
    evl_scores = ["LDA-Aggregated"]
    summary = content_selection.generateSummary(EvaluationData)
    evl_scores.extend(
        rougeScore.computeRougeScore(summary, groundTruthSummary, config['topics'], config['intervals']))
    evaluation.append(evl_scores)
    result_util.saveRougeAnalysis(evaluation, config, fileName)

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
    summary = content_selection.generateSummary(EvaluationData)
    evl_scores.extend(
        rougeScore.computeRougeScore(summary, groundTruthSummary, config['topics'], config['intervals']))
    evaluation.append(evl_scores)
    result_util.saveRougeAnalysis(evaluation, config, fileName)
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
        EvaluationData[count].setZ(lda.Z)
        count +=1
        print("Finishes at: ", datetime.datetime.now())

    print("Topic learning is completed: at", datetime.datetime.now())
    evl_scores = ["JST"]
    summary = content_selection.generateSummary(EvaluationData)
    evl_scores.extend(
        rougeScore.computeRougeScore(summary, groundTruthSummary, config['topics'], config['intervals']))
    evaluation.append(evl_scores)
    result_util.saveRougeAnalysis(evaluation, config, fileName)

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
        EvaluationData[count].setZ(lda.Z)
        count +=1
        print("Finishes at: ", datetime.datetime.now())

    print("Topic learning is completed: at", datetime.datetime.now())
    evl_scores = ["TS"]
    summary = content_selection.generateSummary(EvaluationData)
    evl_scores.extend(
        rougeScore.computeRougeScore(summary, groundTruthSummary, config['topics'], config['intervals']))
    evaluation.append(evl_scores)
    result_util.saveRougeAnalysis(evaluation, config, fileName)

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
        EvaluationData[count].setZ(lda.Z)
        count +=1
        print("Finishes at: ", datetime.datetime.now())

    print("Topic learning is completed: at", datetime.datetime.now())
    evl_scores = ["LDST"]
    summary = content_selection.generateSummary(EvaluationData)
    evl_scores.extend(
        rougeScore.computeRougeScore(summary, groundTruthSummary, config['topics'], config['intervals']))
    evaluation.append(evl_scores)
    result_util.saveRougeAnalysis(evaluation, config, fileName)

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
        EvaluationData[count].setZ(lda.Z)
        count +=1
        print("Finishes at: ", datetime.datetime.now())

    print("Topic learning is completed: at", datetime.datetime.now())
    evl_scores = ["BST"]
    summary = content_selection.generateSummary(EvaluationData)
    evl_scores.extend(
        rougeScore.computeRougeScore(summary, groundTruthSummary, config['topics'], config['intervals']))
    evaluation.append(evl_scores)
    result_util.saveRougeAnalysis(evaluation, config, fileName)


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
        EvaluationData[count].setZ(lda.Z)
        count += 1
        print("Finishes at: ", datetime.datetime.now())

    print("Topic learning is completed: at", datetime.datetime.now())
    evl_scores = ["MJST"]
    summary = content_selection.generateSummary(EvaluationData)
    evl_scores.extend(
        rougeScore.computeRougeScore(summary, groundTruthSummary, config['topics'], config['intervals']))
    evaluation.append(evl_scores)
    result_util.saveRougeAnalysis(evaluation, config, fileName)

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

    print("Topic learning is completed: at", datetime.datetime.now())
    evl_scores = ["MOSAIC"]
    summary = content_selection.generateSummary(EvaluationData)
    evl_scores.extend(rougeScore.computeRougeScore(summary, groundTruthSummary, config['topics'], config['intervals']))
    evaluation.append(evl_scores)
    result_util.saveRougeAnalysis(evaluation, config, fileName)