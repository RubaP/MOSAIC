import rouge
import numpy as np
import src.components.preprocessor as p

import warnings
warnings.filterwarnings('ignore')

def cleanDataForPresentation(text, p):
    text = text.lower()

    cleaned = p.clean(text)
    return cleaned

def compute_rouge(hypothesis, reference):
    evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=3,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    scores = evaluator.get_scores(hypothesis, reference)

    rouge1 = None
    rouge2 = None
    rouge3 = None
    rougeL = None
    rougeW = None

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if metric == 'rouge-l':
            rougeL = results['f']
        elif metric == 'rouge-w':
            rougeW = results['f']
        elif metric == 'rouge-1':
            rouge1 = results['f']
        elif metric == 'rouge-2':
            rouge2 = results['f']
        elif metric == 'rouge-3':
            rouge3 = results['f']

    print("ROUGE-L: ", rougeL)
    print("ROUGE-W: ", rougeW)
    print("ROUGE-1: ", rouge1)
    print("ROUGE-2: ", rouge2)
    print("ROUGE-3: ", rouge3)
    return [rougeL, rougeW, rouge1, rouge2, rouge3]

def computeRougeScore(summary, groundTruthSummaries, K, intervals):
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION)
    results = []

    summary_hypothesis = []

    for i in range(intervals):
        sum = ""

        for j in range(K):
            index = j + i*K
            sum += cleanDataForPresentation(summary[index]['text'][1], p) +". "
        print(sum)
        summary_hypothesis.append(sum)
    print("Hypothesis summary length: ", len(summary_hypothesis))

    for summaryGroundTruth in groundTruthSummaries:
        summary_reference = []
        for i in range(intervals):
            summaryG = summaryGroundTruth[str(i)]

            sum = ""
            for tweet in summaryG:
                sum += tweet.lower() + ". "
            summary_reference.append(sum)

        results.append(compute_rouge(summary_hypothesis, summary_reference))

    print("Average rouge score: ", np.mean(results, axis=0))
    return np.mean(results, axis=0)

def computeRougeScoreBaseline(summary, groundTruthSummaries, K, intervals):
    p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION)
    results = []

    print("Hypothesis summary length: ", len(summary))
    summary_hypothesis = []

    for sum in summary:
        summary_hypothesis.append(p.clean(sum.lower()))

    print("Summary: ", summary_hypothesis)

    for summaryGroundTruth in groundTruthSummaries:
        summary_reference = []
        for i in range(intervals):
            summaryG = summaryGroundTruth[str(i)]

            sum = ""
            for tweet in summaryG:
                sum += tweet.lower() + ". "
            summary_reference.append(sum)

        results.append(compute_rouge(summary_hypothesis, summary_reference))

    print("Average rouge score: ", np.mean(results, axis=0))
    return np.mean(results, axis=0)