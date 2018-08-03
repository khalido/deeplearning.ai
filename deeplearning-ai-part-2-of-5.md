---
title: "deeplearning.ai: Improving Deep Neural Networks"
date: 2018-07-12
tags:
- python
- courses
- deep learning
---

[Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network/home/welcome)

**Course Resources**

- [Discussion forum](https://www.coursera.org/learn/deep-neural-network/discussions)
- [YouTube playlist for Course 1](https://www.youtube.com/watch?v=1waHlpKiNyY&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
- course reviews: [1](https://towardsdatascience.com/review-of-deeplearning-ai-courses-aed1328e4ffe)

## Week 1: Practical aspects of Deep Learning

Train / Dev / Test sets

- training is a highly iterative process as you test ideas and parameters
- even highly skilled practicioners find it impossible to predict the best parameters for a problem, so going through
- split data into train/dev/test sets
  - training set - the actual dataset used to train the model
  - cross validation or dev set - used to evaluate the model
  - test set - only used once the model is finally trained to get a unbiased estimate of the models performance
- depending on the dataset size, use different splits, eg:
  - 100 to 1M ==> 60/20/20
  - 1M to INF ==> 98/1/1 or 99.5/0.25/0.25
- you can mismatch train/test distribution, so make sure they come from the same distribution

Bias / Variance

![](/img/bias-variance.png)

- high bias - model is under fitting, or not even fitting the training set
- high variance - model is over fitting the training and dev set
- you can have both high bias AND high variance
- balance the bias and variance, and also consider the underlying problems tractibility, as are we aiming for 99% accuracy or 70%? Compare with other models and human performance to get a sense of a baseline error

Basic recipe for machine learning

- for high bias errors: bigger NN,more layers, different model, try different activations
- high variance: more data, regularize appropirately, try a different model
- training a bigger NN almost never hurts

Regularization

![](https://upload.wikimedia.org/wikipedia/commons/b/b8/Sparsityl1.png?1531465205841)

- vectors often have many dimensions, so model sizes get BIG. regulariation helps to select features and reduce dimensions, as well as reduce overfitting
- L1 regularization (or Lasso) adds a penalty equal to the sum of absoulte coefficients
- L2 regularization (or Ridge) adds a penalty equal to the sum of squared coefficients

> L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes.

- as a rule of thumb, L2 almost always works better than L1 .

Dropout regularization

- for each iteration, use a probablity to determine whether to "drop" a neuron. So each iteration through the NN drops a different, random set of neurons.
- each layer in the NN can have a different dropout probability, the downside is that we have more hyperparameters to tweak
- in computer vision, we almost always use dropout becuase we are almost always overfitting in vision problems
- a downside of dropout is that the cost function is no longer well defined, so turn off dropout to see that the loss is dropping, which implies that our model is working, then turn on dropout for better training
- remove dropout at test time
- dropout intuitions:
  - can't rely on on any given neuron, so have to spread out weights
  - can work similar to L2 regularization
  - helps prevent overfitting

Data augmentation

- inexpensive way to get more data - e.g with images flip, distory, crop, rotate etc to get more images
- this helps as a regularization technique
-

Early stopping

- stop training the NN when the dev set and training set error start diverging - get the huperparameters from the lowest dev and training set cost.
- this prevents the NN from overfitting on the training set but stops gradient descent early
- Andrew NG generally prefers using L2 regularization instead of early stopping as that


Deep NN suffer from vanishing and exploding gradients

- this happens when gradients become very small or big
- in a deep NN if activations are linear, than the activations and derivates will increase exponentially with layers

Weight Initialization for Deep Networks

- this is a paritial solution to vanishing/exploding gradients
- initialize the weights with a variance equal to 1/n - where n is the number of input features - sure weights are not too small or not to large

Numerical approximation of gradients

Gradient Checking

- check our gradient computation functions - this helps debug backprop
-

## Week 2:Optimizatoni algorithims

Mini-batch gradient descent

- training big data is slow - breaking it up into smaller batches speeds things up
- vectorization allows us to put our entire data set of m examples into a huge matrix and process it all in one go
- but this way we have to process the entire set before our gradient descent can make a small step in the right direction
- training on mini batches allows gradient descent to work much faster, as in one iteration over the dataset we would have taken m / batch_size gradient descent steps
  - makes the leaning more 'noisy' since each mini-batch is new data (compared to going over the entire dataset)
- a typical mini-batch size is 64, 128, 256, 512
  - should be a power of 2 as thats how computer memory is setup
  - make sure the batch fits inside cpu/gpu memory
-

Exponentially weighted averages

- also called exponentially weighted moving averages in statistics
- this decreases the weight of older data exponentially, enhancing the effect of more recent numbers which makes it easy to spot new trends. Of course the parameters can be tweaked, like changing beta depending on how many data points to average`(1 / (1 - beta))`.
- key component of several optimization algos

```V(t) = beta * v(t-1) + (1-beta) * theta(t)```

bias correction in exponentially weighted averages:

- the moving avg in the beginning is low since there isn't past data to refer from and it starts from zero.
- so add a bias term, dividing the above equation by `(1 - beta^t)`:

```v(t) = (beta * v(t-1) + (1-beta) * theta(t)) / (1 - beta^t)```

Gradient descent with momentum

- modify gradient descent so on each iteration compute the exponential weighted averages of the gradients and update weights
- this takes us faster to the min point, and dampens out oscillations
- most common value of beta is `0.9`. generally we don't bother with bias correction since we do so many iterations
- [this explains why momentum works](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d):

> With Stochastic Gradient Descent we don’t compute the exact derivate of our loss function. Instead, we’re estimating it on a small batch. Which means we’re not always going in the optimal direction, because our derivatives are ‘noisy’. Just like in my graphs above. So, exponentially weighed averages can provide us a better estimate which is closer to the actual derivate than our noisy calculations.

- also see [Nesterov Momentum](http://cs231n.github.io/neural-networks-3/#sgd)

RMSprop or Root mean square prop

- we want learning to go fast horizontally and slower vertically - so we divide updates in the vertical direction by a large number and updates in the horizontal direction by a much smaller number
- this dampens oscillations, so we can use a faster learning rate
- first proposed in Hinton's coursera course

Adam optimization algorithim

- mashes together momemtum and RMSprop, works very well
- parameters:
  - learning rate alpha
  - beta1: moving avg or momemtum parameter, same as 0.9 above
  - beta2: RMSprop, `0.999` works well
  - epsilon `10^8` - less important, can leave it at default
  - generally leave values at default, just try out different learning rates

Learning rate decay

- slowly decrease learning rate over epochs
- this makes intuitive sense as in the beginning bigger steps are ok and as the NN starts converging, we need smaller steps
- other methods: exponential decay, discrete steps, etc
- manual decay - watch the model as it trains, pause and manually change the learning rate - works if running a small number of models which take a long time to train
- this is lower down on the list of things to try when tuning

The problem of local optima

- people used to worry a lot about NN getting in local optima, but in multi-dimensional space most points of zero gradient are saddle points so its very unlikely to get stuck in a local optima
- a lot of our intuitions about low dimensional spaces don't transfer over the high dimensional space practically all NN's use - i.e if we have 20K parameters, the NN is operating in a 20K dimensional space
- but plateaus can slow down learning, so techniques like momentum, Adam help here

## Week 3: Hyperparameter tuning, Batch Normalization and Programming Frameworks

Hyperparameter tuning