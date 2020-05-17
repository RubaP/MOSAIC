import datetime
from src.util import  utils as util
from src.components import trendIntervalDetection as timeInterval
import numpy as np
import src.evaluation.util.results as result_util

def computeTotalVolume(bins):
    return np.sum(bins)

def computeVolume(bins, intervals_in_bin_no, totalVolume):
    volume = 0

    for interval in intervals_in_bin_no:

        volume += np.sum(bins[interval[0]: interval[1]+1])
    return volume/totalVolume

print("Started at: ", datetime.datetime.now())
config = util.readConfig()
df = util.readAndProcessFile(config)

min, max = timeInterval.getMinMaxTimne(df)
bins = timeInterval.createBin(df['timestamp'], min, max, config['trend_interval_detection'])
intervalConf = config['trend_interval_detection']
totalVolume = computeTotalVolume(bins)
vol = []
intervals = []
volOPAD = []
intervalsOPAD = []

#Proposed approach
alpha = 0
while alpha < 1.01:
    intervalConf['alpha'] = alpha
    intervals_in_bin_no = timeInterval.findIntervals(bins, intervalConf)
    vol.append(computeVolume(bins, intervals_in_bin_no, totalVolume))
    intervals.append(len(intervals_in_bin_no))
    alpha += 0.025
print("Trend intervals computed for MOSAIC")

#OPAD approach
alpha = 0
while alpha < 1.01:
    intervalConf['alpha'] = alpha
    intervals_in_bin_no = timeInterval.findIntervalsUsingOPAD(bins, intervalConf)
    volOPAD.append(computeVolume(bins, intervals_in_bin_no, totalVolume))
    intervalsOPAD.append(len(intervals_in_bin_no))
    alpha += 0.025
print("Trend intervals computed for OPAD")

res = np.concatenate(([[0 + x*0.025 for x in range(41)]], [vol], [intervals], [volOPAD], [intervalsOPAD]), axis=0)
result_util.saveTrendIntervalDetectionResults(res.T, config, "TrendIntervals")
