import numpy as np
from random import shuffle
import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    exp = np.exp(scores)
    yk = exp[y[i]] / np.sum(exp)
    loss += -math.log(yk)
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:, j] -= (1 - exp[j] / np.sum(exp)) * X[i] # page 209 of PRML
      else:
        dW[:, j] -= (- exp[j] / np.sum(exp)) * X[i]

  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W)
  scores -= np.max(scores, axis = 1).reshape((num_train, 1))
  exp = np.exp(scores)
  normalize = exp / np.sum(exp, axis = 1).reshape((num_train, 1))
  loss = 1.0/num_train * np.sum(-np.log(normalize[np.arange(num_train), y]))
  loss += 0.5 * reg * np.sum(W * W)

  id = np.zeros((num_train, num_classes))
  id[np.arange(num_train), y] = 1
  dW = -1.0/num_train * np.transpose(X).dot(id - normalize)

  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

