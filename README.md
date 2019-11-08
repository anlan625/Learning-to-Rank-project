# Learning-to-Rank-project
Course Project for COMSW4995 Deep Learning, Columbia University

It is widely recognized that the position of an item in the ranking has a crucial influence on its exposure and economic success. However, the algorithm widely used to learn the rankings does not lead to rankings that would be considered fair. The goal is to find a method that not only maximizes utility to the users, but also rigorously enforces merit-based exposure constraints towards the items. In the project, we first apply some conventional methods on LTR(Learning-to-Rank) problems based on SVMRank (Joachims et al., 2009), RankNet (Burges et al., 2005). Furthermore, we recur a new LTR algorithm called Fair-PG-Rank for directly searching the space of fair ranking policies via a policy-gradient approach(Singh and Joachims, 2019). Beyond the theoretical evidence in deriving the framework and the algorithm, we also provide empirical results on simulated and real-world datasets verifying the effectiveness of the approach in individual and group-fairness settings.

In the project, we test the nDCG of the model on Yahoo Dataset.

The files are specified as follow:

SVMRank/code: SVMRank implementation
reference: http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html

SVMRank-model1: the model trained by set1.train by SVMRank

SVMRank-prediction1.txt: the prediction of set1.test by SVMRank-model1

SVMRank-model2: the model trained by set2.train by SVMRank

SVMRank-prediction2.txt: the prediction of set2.test by SVMRank-model2

RankNet-set1.model: the model trained by set1.train by RankNet

RankNet-prediction1.txt: the prediction of set1.test by RankNet-set1.model

RankNet-set2.model: the model trained by set2.train by RankNet

RankNet-prediction2.txt: the prediction of set2.test by RankNet-set2.model

learning2rank: ranking implementation
refernece: https://github.com/shiba24/learning2rank

nDCG.ipynb: the python code use to compute nDCG for predictions

RankNet_main.py: the python code use to train RankNet and compute nDCG for predictions

