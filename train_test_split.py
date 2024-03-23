from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from PIL import Image
from keras.utils import np_utils
# import theano

training = []

# Step 4: Shuffle the Dataset
random.shuffle(training)

# Step 5: Assigning Labels and Features
X =[]
y =[]
for features, label in training:
  X.append(features)
  y.append(label)
X = np.array(X).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3)

# Step 6: Normalizing X and Converting Labels to Categorical Data
X = X.astype('float32') # OpenCV initially stores value as uint8. Floating-point is preferrable for Neural Networks
X /= 255 # X will now be floating point between 0 and 1, easier to categorize
Y = np_utils.to_categorical(y, NUM_CATEGORIES) # One Hot Encoding
print(Y[100])
print(Y.shape)

# Step 7: Split X and Y for Use in CNN
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.25, # 25:75 testing:training ratio
    random_state = NUM_CATEGORIES
)
