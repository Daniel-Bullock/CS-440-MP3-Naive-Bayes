# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP3. You should only modify code
within this file and the last two arguments of line 34 in mp3.py
and if you want-- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from collections import Counter

import numpy as np


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set

    predictions = []

    posFreq = Counter()  # https://www.geeksforgeeks.org/python-count-occurrences-element-list/
    negFreq = Counter()  # https://docs.python.org/2/library/collections.html

    num_pos = 0
    num_neg = 0
    l = len(train_labels)
    for i in range(l):
        if train_labels[i] == 1:
            # num_pos += 1
            posFreq.update(train_set[i])
        else:
            # num_neg += 1
            negFreq.update(train_set[i])
    # print(posFreq)
    pTot = sum(posFreq.values())  # sums all counts of all words up, so as to count all positive words
    individualPos = len(list(posFreq))  # total amount of different words that show up as positive
    nTot = sum(negFreq.values())
    individualNeg = len(list(negFreq))
    # print(pTot)
    # print(individualPos)
    # print(posFreq["the"])   -- total appearances of an individual word such that it's positive
    priorPosDefault = np.log(pos_prior)
    priorNegDefault = np.log(1 - pos_prior)

    for review in dev_set:
        priorPos = priorPosDefault
        priorNeg = priorNegDefault
        for word in review:
            if posFreq[word] == 0:
                num_pos += 1
                num_neg += 1
            temp_p = ((posFreq[word] + smoothing_parameter) / (smoothing_parameter * individualPos + pTot))
            temp_n = ((negFreq[word] + smoothing_parameter) / (smoothing_parameter * individualNeg + nTot))
            priorPos += np.log(temp_p)
            priorNeg += np.log(temp_n)
        if priorPos > priorNeg:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions


def bigramBayes(train_set, train_labels, dev_set, unigram_smoothing_parameter=1.0, bigram_smoothing_parameter=1.0,
                bigram_lambda=0.5, pos_prior=0.8):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    unigram_smoothing_parameter - The smoothing parameter for unigram model (same as above) --laplace (1.0 by default)
    bigram_smoothing_parameter - The smoothing parameter for bigram model (1.0 by default)
    bigram_lambda - Determines what fraction of your prediction is from the bigram model and what fraction is from the unigram model. Default is 0.5
    pos_prior - The prior probability that a word is positive. You do not need to change this value.
    """
    # TODO: Write your code here
    # return predicted labels of development set using a bigram model
    predictions = []

    posFreq = Counter()  # https://www.geeksforgeeks.org/python-count-occurrences-element-list/
    negFreq = Counter()  # https://docs.python.org/2/library/collections.html

    num_pos = 0
    num_neg = 0
    l = len(train_labels)
    for i in range(l):
        if train_labels[i] == 1:
            # num_pos += 1
            posFreq.update(train_set[i])
        else:
            # num_neg += 1
            negFreq.update(train_set[i])
    # print(posFreq)
    pTot = sum(posFreq.values())  # sums all counts of all words up, so as to count all positive words
    individualPos = len(list(posFreq))  # total amount of different words that show up as positive
    nTot = sum(negFreq.values())
    individualNeg = len(list(negFreq))
    # print(pTot)
    # print(individualPos)
    # print(posFreq["the"])   -- total appearances of an individual word such that it's positive
    priorPosDefault = np.log(pos_prior)
    priorNegDefault = np.log(1 - pos_prior)

    for review in dev_set:
        priorPos = priorPosDefault
        priorNeg = priorNegDefault
        for word in review:
            if posFreq[word] == 0:
                num_pos += 1
                num_neg += 1
            temp_p = ((posFreq[word] + unigram_smoothing_parameter) / (unigram_smoothing_parameter * individualPos + pTot))
            temp_n = ((negFreq[word] + unigram_smoothing_parameter) / (unigram_smoothing_parameter * individualNeg + nTot))
            priorPos += np.log(temp_p)
            priorNeg += np.log(temp_n)
        if priorPos > priorNeg:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions
