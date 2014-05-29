import numpy 
import theano.tensor as T
import theano 

from HiddenLayer import HiddenLayer
from lr import LogisticRegression

class multiLayerMLP(object):
  def __init__(self, rng, input, n_in, n_hidden, n_out):
    self.hiddenLayer1 = HiddenLayer(rng=rng, input = input, n_in = n_in, n_out = n_hidden, activation = T.tanh)
    self.hiddenLayer2 = HiddenLayer(rng=rng, input = self.hiddenLayer1.output, n_in = n_hidden, n_out = n_hidden, activation = T.tanh)
    self.logRegressionLayer = LogisticRegression(input = self.hiddenLayer2.output, n_in = n_hidden, n_out = n_out)
    self.L1 = abs(self.hiddenLayer1.W).sum() + abs(self.hiddenLayer2.W).sum() +  abs(self.logRegressionLayer.W).sum()
    self.L2_sqr = (self.hiddenLayer1.W ** 2).sum() + (self.hiddenLayer2.W **2).sum() +  (self.logRegressionLayer.W **2).sum()
    self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
    self.errors = self.logRegressionLayer.errors
    self.params = self.hiddenLayer1.params + self.hiddenLayer2.params +  self.logRegressionLayer.params
