---
title: "deeplearning.ai: Structuring Machine Learning Projects"
date: 2018-08-07
cover: /img/deeplearningai-2-of-5-cert.png
tags:
- courses
- deep learning
---

> You will learn how to build a successful machine learning project. If you aspire to be a technical leader in AI, and know how to set direction for your team's work, this course will show you how.


**Course Resources**

- [Course home](https://www.coursera.org/learn/machine-learning-projects/home/welcome)
- [Discussion forum](https://www.coursera.org/learn/machine-learning-projects/discussions)

## Week 1: ML Strategy

There are lots of ways to improve a DL system, so its important to sse quick and effective ways to figure out the most promising things to try and and improve.

Orthogonalization

- old school tv's had a number of knobs to tune the picture - all the settings made it very hard to get the picture perfect
- cars have orthogonal controls - a steering and speed which effect two different things making it easier to control. If we have controls which effected both steering and speed at the same time it would make it much harder to get a speed and steering angle we wanted.
- we have a chain of assumptions in ML: fit training set well on cost func, fit dev set, fit test set, then perform well in the real world and we want to have different set of knobs to tune each part.
- ideally we have a number of controls which do one task well without effecting other things too much. 
- of course, some controls apply across many layers and are still useful, like early stopping

### Setting up your goal

Single number evaluation metric

- set up a single real number to evaluate performance before starting out on a project
- a well defined dev set and a single metric speeds up the iterative process of devloping a good model
- looking at a simple cat classifier

| Classifier | Precision | Recall |
| ---------- | --------- | ------ |
| A          | 95%       | 90%    |
| B          | 98%       | 85%    |

- **Precision**: of the examples our classifier says are cats, what percentage is actually right
- **Recall**:  what % of actual cats are correctly classified
- The problem with using precision and recall is it can make it hard to figure out which classifier is better
- [F1 Score](https://www.wikiwand.com/en/F1_score) is the harmonic average of precision and recall using $F1 = 2 / ((1/P) + (1/R))$ or $2 * (precision * recall) / (precision + recall)$
    - F1 score is usefeul as it balances precision and recall, rather than just taking the simple average, which would favour outliers
- to summarize, have a single number metric (this could be an average of several metrics) and a good dev set

Satisficing and Optimizing metric

- its not easy to setup a single number - we might narrow things down to a single number, accuracy, but also want to evaluate running time
- solve this by picking one metric to optimize, and satisfy the other metric by picking a threshold, like all running times below 1000ms are good enough.
    - for voice recognition, like on Amazon Alexa, we have metrics for how long humans are ok to wait, so we can just use that number as a threshold
- - so, we have a make metric to optimize, and any number of other metrics which we satisfy with thresholds. 
 
 Train/dev/test distributions

 - dev (cross validation set) and test (the final holdout data) sets have to come from the same distribution.
 - for example, if we have data from different regions, don't say us/uk/europe is the dev set and asia is the test set - regions will have differences.
 - an ML team wasted time optimizing loan approvals on medium income zip codes, but was testing on low income zip codes
 - choose a dev and test set to reflect data you expect to get in the future and consider important to do well on

Size of the dev and test sets

- old rule of thumb for train/dev/test: 60/20/20 - worked for smaller data
- but now we have much larger data sets, so for 1M data set, 1% or 10K might be enough for a test set.
- deep learning is data hungry so we want to feed it as much data as possible for training
- the test set helps us evaluate the overall system, so it has to be big enough to give us high confidence.
    - this will vary depending on the data set
    - sometimes we might not need a test set at all

When to change dev/test sets and metrics

- say we've built two cat classifiers, A has 3% error, B has 5% error
- A is better, but lets through porn images. B stops all the porn images, so even though it has higher error, B is better algorithm.
- so we want to change our metrics here, say adding a weight to penalize porn images
- think of machine learning as having two separate steps:
    - 1. figure out the metric
    - 2. worry about how to actually do well on the metric
- other things happen, like our image classifier performs well on our data set, but users upload bad quality pictures which lower performance, so change metric and/or the dev set to better capture what we need our algorithm to actually do

### Comparing to human-level performance

Why human-level performance?

- two main reasons we compare to human level performance:
    - ML is getting better so its become feasible to compare to human level performance in many applications
    - the workflow of designing and building ML systems is much more efficient when comparing to what humans also do
- for many problems, progress is rapid approaching human level performance, and slows down upon reaching it. the hope is that its reaching a theoretical limit, called the Bayes optimal error.
- **Bayes optimal error** is the best possible error
    - for tasks humans are good at, there isn't much range b/w human error and the bayes optimal error.
- when our ML algo is worse than humans, get humans to:
    - give us more data, like label images, translate text, etc.
    - manual analysis errors - why did a human get it right and algo get it wrong?
    - better analysis of bias/variance
- 
Avoidable bias

- humans are great at image classification, so looking at two image classification algos:

|                | Bias Example | Variance Example |
| -------------- | ------------ | ---------------- |
| Humans         | 1%           | 7.5%             |
| Training error | 8%           | 8%               |
| Dev Error      | 10%          | 10%              |

- the left example illustrates  bias - since humans are doing much better than the algo, we focus on improving training error
- the right example shows that we are close to human error - so we focus on reducing the variance
- if we are doing better than Bayes error we're overfitting
- having an estimate of the bayes optimal error helps us to focus on whether to reduce bias or variance.
    - avoidable bias: training error - human level error
    - variance: dev error - training error

Understanding human-level performance

- there are multiple levels of human performance - choose what matters and what we want to achieve for our system
    - do we use a typical doctors error level, an experienced doctors, or the lower error level from a team of experienced doctors.
    - surpassing an average radiologist might mean our system is good enough to deploy
- use human level error as a proxy for Bayes error. 
- An error analysis example from medical classification, where say human error ranges from 0.5-1%:

| Error       | Bias Example | Variance Example |
| ----------- | ------------ | ---------------- |
| Human/Bayes | 1%           | 1%               |
| Training    | 5%           | 1%               |
| Dev         | 6%           | 5%               |

- the left column we need to concentrate on reducing the training error - the variance, and the human error we pick doesn't matter.
- the right col we need to concentrate on the dev error - i.e the variance, and picking the lower range of human error is more important since we are close to human level performance.
- so we have a more nuanced view of error, and using bayes error instead of zero error leads to better and faster decisions on reducing bias or variance.

Surpassing human-level performance

- ml gets harder as we approach/surpass human level performance
- when we've surpassed human level performance, are we overfitting, or is the bayes error lower than human error
- some applications have surpassed human level: online advertising, product recommendation, loan approval, logistics
    - all these examples are learning from structured data, not natural perception problems which humans are very good at
    - teams have access to heaps of data
- computers haven gotten to single human level at certain perception tasks, like image recognition, speech.

Improving your model performance

- two fundamental assumptions of supervised learning:
    - we can fit the training set pretty well, or avoid bias
    - the training set performance generalizes well to the dev/test set - achieve low variance
- to improve a deep learning supervised system
- for improving bias:
    - train bigger model
    - train longer or use better optimization algo (RMSprop, Adam, momentum, etc)
    - try other model architectures, hyperparameter search
- improve variance:
    - more data
    - regularization (L2, dropout, data augmentation)
    - NN architecture, hyperparameter search
- 

### Andrej Karpathy interview

- liked neural networks / machine learning where humans write the optimization algo and the computers write code
- while working on cifar10, had predicted an error rate of 10%, but we're now down to 2-3%.
- built a javascript interface to show himself imagenet images and classify them
- famous for putting his deep learning stanford class online
- understand of deep learning has changed:
    - surprised by how general deep learning has - no one saw how well it would work
    - people are crushing 