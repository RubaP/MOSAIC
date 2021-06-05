### MOSAIC
This code contains the implementation of the project, MOSAIC that generates an emotion-aware event summary from microblogs related to an event. Check the following links for more details.

Conference Paper: This work has been presented at SocialNLP 2021 which held in conjuction with TheWebConf 2021. Please refer to the following link for the full paper.
https://dl.acm.org/doi/10.1145/3442442.3452311

Thesis: This work has been done as a Master Thesis in School of Computing, National University of Singapore. Please refer to the following link for the thesis.
https://scholarbank.nus.edu.sg/handle/10635/188077


###### Required packages
* tweepy
* nltk
* pandas
* numpy
* sklearn
* emoji
* twikenizer
* gensim

###### Collect tweets
* Update twitter API credentials and file location to tweetId in DataCollectionScripts/twitter.py
* Run DataCollectionScripts/twitter.py to collect the tweets
* This script saves the necessary information required to run the project as a csv file and also saves the tweet objects as json which is not required to run the project

###### Evaluation Metrics
* Evaluation metrics can be found at src/evaluation/metrics
* All the metrics scripts accept list of src/evaluation/Data objects each representing an interval with information learnt by the models e.g. word dis, emotion dis etc.
* coherence.py computes the average PMI of top 5,10, and 20 words of given topics learned
* KLDivergence.py computes the average KL-Divergence of the emotion distributions of all the topics learned
* rougeScore.py computes rouge value for given reference summary and model summary

###### Running evaluation scripts
* update config.json
* run src/evaluation/PMISensitivityAnalysis.py to get PMI value of top 5 words for all the models for range of topics
* run src/evaluation/TrendIntervalDetection.py to compute to average intervals for range of alpha
* run src/evaluation/KLDivergence.py to compute KL-Divergence of emotion distribution of all the models
* run src/evaluation/RougeEvaluation.py to compute average rouge value against three reference summaries for all the models
