# USAGE
# python nn_mnist.py

# import the necessary packages
from submodules.nn import NeuralNetwork
from sklearn.preprocessing import LabelBinarizer # performing one-hot encoding
from sklearn.model_selection import train_test_split # create train/test data
from sklearn.metrics import classification_report # display performance
from sklearn import datasets # access to mnist dataset

# load the MNIST dataset and apply min/max scaling to scale the
# pixel intensity values to the range [0, 1] (each image is
# represented by an 8 x 8 = 64-dim feature vector)
print("[INFO] loading MNIST (sample) dataset...")
digits = datasets.load_digits() # loads a sample of the mnist dataset
data = digits.data.astype("float") # convert from int to floatingpoint
data = (data - data.min()) / (data.max() - data.min()) # scale data from 0 to 1 using min/max scaling
print("[INFO] samples: {}, dim: {}".format(data.shape[0],
	data.shape[1])) # displays total number of samples

# construct the training and testing splits
(trainX, testX, trainY, testY) = train_test_split(data,
	digits.target, test_size=0.25)

# convert the labels from integers to vectors
# one-hot encoding on labels
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# train the network
print("[INFO] training network...")
# input is 64, 32 hidden layer, 16 node hidden layer, 10 node hidden layer
# mnist has 0 to 9 values, so we set to 10 possible answers
nn = NeuralNetwork([trainX.shape[1], 32, 16, 10])
print("[INFO] {}".format(nn))
nn.fit(trainX, trainY, epochs=1000)

# evaluate the network
print("[INFO] evaluating network...")
predictions = nn.predict(testX)
predictions = predictions.argmax(axis=1)
print(classification_report(testY.argmax(axis=1), predictions))