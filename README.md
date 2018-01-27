# Sentiment-Analysis
A Sentiment Analyzer for a set of Hotel Reviews using Naive Bayes Algorithm

Implementation of Basic Naïve Bayes Algorithm for the sentiment analysis of the positive and negative reviews of the many hotels given in the Training Set. A Unigram Language Model is built for each class – Positive and Negative Classes.

As a first step, Regular Expressions are used to clean up the data, that is, to eliminate the punctuation marks - !,?,-,. Etc and to eliminate frequently repeated words called Stop Words, like ‘the’,’a’ etc., the Stop Word dictionary from nltk package was used. Laplace Smoothing is done to avoid zero probabilities appearing for likelihood. Likelihood probability for each word is calculated as a ratio of  frequency of each word to the sum of frequency of all other words in the class and size of the vocabulary for that class. For calculating the Prior P(c), the fraction of number of reviews in each class to the sum of number of reviews in both the classes is considered. To avoid underflow, all the probabilities are calculated with log to the base 10. And unknown words are ignored. Finally for test set, the probability for each review is calculated for each class and whichever class has the highest proabability value, the review is assigned that particular class. The class being POS and NEG here representing Positive and Negative Reviews. 

The given training set is divided in the ratio of 80:20 for training and test set. The accuracy achieved was around 88% using the Basic Naives Approach. 
My second approach is using naïve bayes library of textblob for higher accuracy. With this library, the accuracy achieved was around 90%.
The output file from the second approach is attached here. 
