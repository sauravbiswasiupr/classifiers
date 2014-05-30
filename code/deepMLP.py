import numpy 
import theano.tensor as T
import theano 

from HiddenLayer import HiddenLayer
from lr import LogisticRegression

class deepMLP(object):
  def __init__(self, rng, input, n_in, n_hidden, n_out):
    self.hiddenLayers = []
    initialHiddenLayer = HiddenLayer(rng=rng, input = input, n_in = n_in, n_out = n_hidden, activation = T.tanh)
    self.hiddenLayers.append(initialHiddenLayer)
    for i in range(1, 6):
      hiddenLayer = HiddenLayer(rng=rng, input = self.hiddenLayers[i-1].output, n_in = n_hidden, n_out = n_hidden, activation = T.tanh)
      self.hiddenLayers.append(hiddenLayer)
    
    self.logRegressionLayer = LogisticRegression(input = self.hiddenLayers[-1].output, n_in = n_hidden, n_out = n_out)
    self.L1 = abs(self.logRegressionLayer.W).sum()
    self.L2_sqr = (self.logRegressionLayer.W **2).sum()

    for layer in self.hiddenLayers:
      self.L1 += abs(layer.W).sum()
      self.L2_sqr += (layer.W **2).sum()

    self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
    self.errors = self.logRegressionLayer.errors
    self.params = self.logRegressionLayer.params
    for layer in self.hiddenLayers:
      self.params += layer.params
