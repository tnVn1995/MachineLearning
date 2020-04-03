# Sentiment_Classifier

Train a Logistic Regression classifier to recognize positive/negative comments from the IMDb dataset.
Using SGDClassifier to perform online learning to deploy the classifier on PythonEverywhere.

## [Data](http://ai.stanford.edu/~amaas/data/sentiment/)

This dataset contains movie reviews from the Internet Movie Database (IMDb) that was used in 
Publications Using the Dataset
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

As described in the data homepage:

> This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. 

## Requirement

- [Anaconda](https://www.anaconda.com/distribution/#download-section)

## Intructions

- Download and install Anaconda
- In order to recreate the notebook results, clone the repo to your local machine.
- Move to the directory that contains the repo from an anaconda command line
- Type in conda env create -f environment.yml
- Once done, type conda activate sentimentclf. Then launch jupyter notebook to recreate the notebook.

## Demo

- Here's a [demo](http://tung2921.pythonanywhere.com/results) of the trained classifier hosted on PythonEverywhere  