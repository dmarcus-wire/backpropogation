# import the necessary packages
import numpy as np


class NeuralNetwork:
	# specify list of layers (e.g 2-2-1 or 3-3-1)
	# accept 2 input nodes, 2 hidden layers, 1 output
	# alpha is the learning rate
	def __init__(self, layers, alpha=0.1):
		# initialize the list of weights matrices, then store the
		# network architecture and learning rate
		# think of each layer as a matrix
		# during backpropation we will update this matrix
		self.W = []
		# store layer architecture
		self.layers = layers
		# store learning rate
		self.alpha = alpha

		# start looping from the index of the first layer but
		# stop before we reach the last two layers,
		# because we have to init final 2 layers
		for i in np.arange(0, len(layers) - 2):
			# randomly initialize a weight matrix connecting the
			# number of nodes in each respective layer together,
			# adding an extra node for the bias
			# calc the ith + ith+1 layer for the bias trick
			# 2x2 > 3x3 >
			w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
			self.W.append(w / np.sqrt(layers[i]))

		# the last two layers are a special case where the input
		# connections need a bias term but the output does not
		# drop the +1 for bias
		w = np.random.randn(layers[-2] + 1, layers[-1])
		self.W.append(w / np.sqrt(layers[-2]))

	def __repr__(self):
		# construct and return a string that represents the network
		# architecture; simply print statement of layer names
		return "NeuralNetwork: {}".format(
			"-".join(str(l) for l in self.layers))

	def sigmoid(self, x):
		# compute and return the sigmoid activation value for a
		# given input value
		return 1.0 / (1 + np.exp(-x))

	def sigmoid_deriv(self, x):
		# compute the derivative of the sigmoid function ASSUMING
		# that `x` has already been passed through the `sigmoid`
		# function
		return x * (1 - x)

	# require paramts:
	# X = bitcoin price, stock, pixels, etc. "whats coming in"
	# y = target output predictions e.g names of labels in MNIST
	# train for number of epochs
	# display to terminal to observe
	def fit(self, X, y, epochs=1000, displayUpdate=100):
		# insert a column of 1's as the last entry in the feature
		# matrix -- this little trick allows us to treat the bias
		# as a trainable parameter within the weight matrix
		X = np.c_[X, np.ones((X.shape[0]))]

		# loop over the desired number of epochs
		for epoch in np.arange(0, epochs):
			# loop over each individual data point and train
			# our network on it
			for (x, target) in zip(X, y):
				# performs that actual prediction, feedforward, calc error and loss
				# x = current data point
				# target = labelled value
				self.fit_partial(x, target)

			# check to see if we should display a training update
			if epoch == 0 or (epoch + 1) % displayUpdate == 0:
				loss = self.calculate_loss(X, y)
				# want this to decrease
				print("[INFO] epoch={}, loss={:.7f}".format(
					epoch + 1, loss))

	# this is where backpropagation lives "feed forward stage"
	# meat of the plate
	# accepts input x
	# returns labeled output y
	def fit_partial(self, x, y):
		# construct our list of output activations for each layer
		# as our data point flows through the network; the first
		# activation is a special case -- it's just the input
		# feature vector itself
		# 2D matrix, for each input x, [] wraps in a list
		A = [np.atleast_2d(x)]

		# FEEDFORWARD:
		# loop over the layers in the network
		# from input 0 to length of the weight matricies
		for layer in np.arange(0, len(self.W)):
			# feedforward the activation at the current layer by
			# taking the dot product between the activation and
			# the weight matrix -- this is called the "net output"
			# to the current layer
			net = A[layer].dot(self.W[layer])

			# computing the "net output" is simply applying our
			# non-linear activation function to the net input
			out = self.sigmoid(net)

			# once we have the net output, add it to our list of
			# activations
			A.append(out)

		# BACKPROPAGATION
		# the first phase of backpropagation is to compute the
		# difference between our *prediction* (the final output
		# activation in the activations list) and the true target
		# value
		error = A[-1] - y

		# from here, we need to apply the chain rule and build our
		# list of deltas `D`; the first entry in the deltas is
		# simply the error of the output layer times the derivative
		# of our activation function for the output value
		# take error x sig derivative, basically calling the final node in the list end-1
		D = [error * self.sigmoid_deriv(A[-1])]

		# once you understand the chain rule it becomes super easy
		# to implement with a `for` loop -- simply loop over the
		# layers in reverse order (ignoring the last two since we
		# already have taken them into account)
		for layer in np.arange(len(A) - 2, 0, -1):
			# the delta for the current layer is equal to the delta
			# of the *previous layer* dotted with the weight matrix
			# of the current layer, followed by multiplying the delta
			# by the derivative of the non-linear activation function
			# for the activations of the current layer
			delta = D[-1].dot(self.W[layer].T)
			delta = delta * self.sigmoid_deriv(A[layer])
			D.append(delta)

		# since we looped over our layers in reverse order we need to
		# reverse the deltas ::-1 means reverse order in python
		D = D[::-1]

		# WEIGHT UPDATE PHASE
		# loop over the layers
		for layer in np.arange(0, len(self.W)):
			# update our weights by taking the dot product of the layer
			# activations with their respective deltas, then multiplying
			# this value by some small learning rate and adding to our
			# weight matrix -- this is where the actual "learning" takes
			# place
			self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

	# accept input X
	# boolean if we need to add bias
	def predict(self, X, addBias=True):
		# initialize the output prediction as the input features -- this
		# value will be (forward) propagated through the network to
		# obtain the final prediction
		p = np.atleast_2d(X)

		# check to see if the bias column should be added
		if addBias:
			# insert a column of 1's as the last entry in the feature
			# matrix (bias)
			p = np.c_[p, np.ones((p.shape[0]))]

		# loop over our layers in the network
		for layer in np.arange(0, len(self.W)):
			# computing the output prediction is as simple as taking
			# the dot product between the current activation value `p`
			# and the weight matrix associated with the current layer,
			# then passing this value through a non-linear activation
			# function
			p = self.sigmoid(np.dot(p, self.W[layer]))

		# return the predicted value
		return p

	# how good/bad we are at making predictions
	# Mean Squared Error
	def calculate_loss(self, X, targets):
		# make predictions for the input data points then compute
		# the loss
		targets = np.atleast_2d(targets)
		predictions = self.predict(X, addBias=False)
		# diff between predicted and targets
		loss = 0.5 * np.sum((predictions - targets) ** 2)

		# return the loss
		return loss