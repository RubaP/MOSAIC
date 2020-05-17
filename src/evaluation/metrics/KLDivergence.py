from scipy.stats import entropy
import numpy as np

"""
Compute average KL divergence between emotion distributions within the interval
"""
def compute_divergence(lda_list):
    divergence = []
    intervals = len(lda_list)
    for interval in range(intervals):
        lda = lda_list[interval]
        print(lda.phiE)
        print("=======================")
        for k in range(lda.K):
            for p in range(lda.K):
                if p != k:
                    divergence.append(entropy(lda.phiE[k], lda.phiE[p]))
    print("Average Divergence of Emotion Distribution: ", np.mean(divergence))
    return np.mean(divergence)