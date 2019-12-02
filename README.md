# Learning-to-Rank-project
Course Project for COMSW4995 Deep Learning, Columbia University

It is widely recognized that the position of an item in the ranking has a crucial influence on its exposure and economic success. However, the algorithm widely used to learn the rankings does not lead to rankings that would be considered fair. The goal is to find a method that not only maximizes utility to the users, but also rigorously enforces merit-based exposure constraints towards the items. In the project, we first apply some conventional methods on LTR(Learning-to-Rank) problems based on SVMRank (Joachims et al., 2009), RankNet (Burges et al., 2005). Furthermore, we use a meta model to stack SVMRank and RankNet to enhance the result. Beyond the theoretical evidence in deriving the framework and the algorithm, we also provide empirical results on simulated and real-world datasets verifying the effectiveness of the approach in individual and group-fairness settings.

The files are specified as follow:

**Output**:  
Prediction and saved models are in the folder SVMRank and RankNet:<br/>
SVMRank-model1: the model trained by set1.train by SVMRank<br/>
SVMRank-model2: the model trained by set2.train by SVMRank<br/>
SVMRank-prediction1.txt: the prediction of set1.test by SVMRank-model1<br/>
SVMRank-prediction2.txt: the prediction of set2.test by SVMRank-model2<br/>
RankNet-set1.model: the model trained by set1.train by RankNet<br/>
RankNet-set2.model: the model trained by set2.train by RankNet<br/>
RankNet-prediction1.txt: the prediction of set1.test by RankNet-set1.model<br/>
RankNet-prediction2.txt: the prediction of set2.test by RankNet-set2.model<br/>

**Running Code**:<br/>
nDCG.ipynb: the python code used to compute nDCG for predictions<br/>
RankNet_main.py: the python code used to train RankNet and compute nDCG for predictions<br/>
metadata.py: the python code used to generate meta data for stack model<br/>
stack.py: the python code used to read in meta data of SVMRank and RankNet and train stacked model<br/>

**Reference**:<br/>
SVMRank/code: SVMRank implementation Reference: http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html<br/>
learning2rank: ranking implementation Refernece: https://github.com/shiba24/learning2rank<br/>
