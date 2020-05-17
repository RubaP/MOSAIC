import pandas as pd

def saveEmDisDivergence(evaluation, config, name):
    df = pd.DataFrame(evaluation)
    df.to_csv(config['event_name'] + '_'+name+'.csv', index=False,
              header=["Variation", "Divergence"])

def saveEvaluations(evaluation, config, name):
    df = pd.DataFrame(evaluation)
    df.to_csv(config['event_name'] + '_'+name+'.csv', index=False, header=["Variation", "PMI5", "PMI10", "PMI20"])

def saveSensitivityAnalysis(evaluation, config, name):
    df = pd.DataFrame(evaluation)
    df.to_csv(config['event_name'] + '_'+name+'.csv', index=False, header=["Variation", "K", "PMI5", "PMI10", "PMI20"])

def saveRougeAnalysis(evaluation, config, name):
    df = pd.DataFrame(evaluation)
    df.to_csv(config['event_name'] + '_' + name + '.csv', index=False,
              header=["Variation", "ROUGE-L", "ROUGE-W", "ROUGE-1", "ROUGE-2", "ROUGE-3"])

def saveTrendIntervalDetectionResults(evaluation, config, name):
    df = pd.DataFrame(evaluation)
    df.to_csv(config['event_name'] + '_'+name+'.csv', index=False,
              header=["Alpha", "Volume_MOSAIC", "Intervals_MOSAIC", "Volume_OPAD", "Intevral_OPAD"])