import time
import numpy as np
import pandas as pd #THIS IS REQUIRED TO SAVE TWEET RATE
"""
Fine intervals using the popular OPAD algorithm
"""
def findIntervalsUsingOPAD(C, config):
    var_threshold = config['var_threshold']
    intervals = []
    mean = np.mean(C)
    variance = np.var(np.asarray(C[:5]))
    print("======STARTED FINDING INTERVALS======")
    i=1
    while i < len(C):
        if ((C[i] - mean)/variance) > var_threshold and C[i] > C[i-1]:
            start = i - 1
            while i < len(C) and C[i] > C[i-1]:
                mean, variance = update(mean, variance, C[i], config)
                i = i+1
            while i < len(C) and C[i] >= C[start]:

                if ((C[i] - mean) / variance) > var_threshold and C[i] > C[i - 1]:
                    end = i - 1
                    break
                else:
                    mean, variance = update(mean, variance, C[i], config)
                    i = i+1
                    end = i
            if i == len(C):
                i = i -1
            if C[i] < C[start]:
                end = i  - 1
            intervals.append((start, end))
        else:
            mean, variance = update(mean, variance, C[i], config)
        i = i + 1
    print("Intervals identified: ", intervals)
    print("=====FINDING INTERVALS COMPLETED=====")
    return intervals

"""
Fine intervals using the proposed peak detection algorithm
"""
def findIntervals(C, config):
    var_threshold = config['var_threshold']
    globalMean = np.average(C)
    intervals = []
    mean = C[0]

    variance = np.var(np.asarray(C[0:5]))
    print("======STARTED FINDING INTERVALS======")
    i=1
    while i < len(C):
        if True: #((C[i] - mean)/variance) > var_threshold and C[i] > C[i-1]:
            start = i - 1
            while i < len(C) and C[i] > C[i-1]:
                mean, variance = update(mean, variance, C[i], config)
                i = i+1
            while i < len(C):# and C[i] >= C[start]:
                if ((C[i] - mean) / variance) > var_threshold and C[i] > C[i - 1]:
                    i = i -1
                    break
                else:
                    mean, variance = update(mean, variance, C[i], config)
                    i = i+1
            #To be checked
            if i == len(C):
                i -=1
            mergeIntervals(C, intervals, start, i, globalMean, config)
            i = i + 2
        else:
            mean, variance = update(mean, variance, C[i], config)
            i = i + 1
    print("Intervals identified: ", intervals)
    print("=====FINDING INTERVALS COMPLETED=====")
    return intervals

"""
Get intervals given the dataset
"""
def getIntervalsFromDataset(df, config):
    min, max = getMinMaxTimne(df)
    bins = createBin(df['timestamp'], min, max, config)
    if config['algorithm'] == 'MOSAIC':
        intervals_in_bin_no = findIntervals(bins, config)
    elif config['algorithm'] == 'OPAD':
        intervals_in_bin_no = findIntervalsUsingOPAD(bins, config)
    else:
        intervals_in_bin_no = None
    intervals = convertIntervalNoToTimestamp(intervals_in_bin_no, min, config)
    return intervals

"""
Get min and max timestamp of a given dataset
"""
def getMinMaxTimne(df):
    min = df['timestamp'].min()
    max = df['timestamp'].max()
    print("Minimum time: ", time.gmtime(min))
    print("Maximum time: ", time.gmtime(max))
    return  min, max

"""
Creates bin to execute trend interval detection
"""
def createBin(timeCol, min, max, config):
    hours = config['interval']
    noOfBins = int((max - min)/(3600*hours)) + 1
    rate = np.zeros(noOfBins)
    print("Bins initialized with size", noOfBins)

    for time in timeCol:
        index = int((time - min)/(3600*hours))
        rate[index] = rate[index] + 1

    """Save tweet rate if necessary"""
    #columns = ['rate']
    #rate_data = pd.DataFrame(rate, columns=columns)
    #rate_data.to_csv("tweet_rate_huricane.csv", index=True)
    return rate

"""
Update mean and variance in trend interval detection algorithm
"""
def update(E, V, N, config):
    alpha = config['alpha']
    Diff = E - N
    var = alpha * Diff + (1 - alpha) * V
    mean = alpha * N + (1 - alpha) * E
    return mean, var

def consecutive_zero(data):
    longest = 0
    current = 0
    for num in data:
        if num == 0:
            current += 1
        else:
            longest = max(longest, current)
            current = 0

    return max(longest, current)

"""
Merge operation of trend interval detection algorithm
"""
def mergeIntervals(bins, intervals, start, end, globalMean, config):
    burstiness_min = config['burstiness_min']
    maxIntervalSize = config['maxIntervalSize']
    burstiness_max = config['burstiness_max']
    if start == 0:
        intervals.append((start, end))
        return intervals

    width = end - start + 1
    avg = np.sum(bins[start: end + 1])/width
    previousStart, previousEnd = intervals[-1]
    previousWidth = previousEnd - previousStart +1
    previousavg = np.sum(bins[previousStart: previousEnd + 1]) / previousWidth


    if(consecutive_zero(bins[previousStart: previousEnd + 1]) >= maxIntervalSize): #Not necessary genera;;y
        intervals.append((start, end))
    elif (np.sum(bins[start: end + 1]) < width * globalMean * burstiness_min):
        if (previousavg/avg > config['burstiness_max']) or (np.sum(bins[previousStart: previousEnd+ 1]) < maxIntervalSize * globalMean * burstiness_max):
            intervals[-1] = (previousStart, end)
        else:
            intervals.append((start, end))
    elif (np.sum(bins[previousStart: previousEnd + 1]) < previousWidth * globalMean * burstiness_min):
        if (avg/previousavg > config['burstiness_max']) or np.sum(bins[start: end + 1]) < maxIntervalSize * globalMean * burstiness_max:
            intervals[-1] = (previousStart, end)
        else:
            intervals.append((start, end))
    else:
        intervals.append((start, end))


    return intervals

"""
Convert bin number to timestamp back
"""
def convertIntervalNoToTimestamp(bins, min, config):
    converted = []
    intervalInMS = config['window_size']*3600
    for bin in bins:
        converted.append((bin[0]*intervalInMS + min, bin[1]*intervalInMS + min))
    return converted
