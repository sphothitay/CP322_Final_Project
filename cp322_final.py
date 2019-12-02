import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Activation
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools

# Loads the mnist dataset (Details in report)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize tensors by scaling input values to type float32 that have values 0 or 1
x_train = X_train.astype("float32")
x_test = X_test.astype("float32")
x_train /= 255
x_test /= 255

# Need vector array of 784 numbers for NN input
x_train = X_train.reshape(60000, 784)
x_test = X_test.reshape(10000, 784)

# Sets up categorical representation of the digit class where the indexes are the class values and an element 1 indicates that it was classified as that index value
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Initialize the ANN with an initial layer and a hidden layer. Dense means fully connected
model = Sequential()
model.add(Dense(10, activation='sigmoid', input_shape=(784,))) # Input as a tensor that has 784 features
model.add(Dense(10, activation='softmax')) # 10 neurons that will result in a matrix of 10 probabilities for the digit classes

# Display summary of model
model.summary()

# Initialize ANN with hyper parameters
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# Train ANN using set batch size and epochs
model.fit(x_train, y_train, batch_size=100, epochs=10)

# Predict classes for test data
test_loss, test_acc = model.evaluate(x_test, y_test)

# Product digit probabilities
digit_probabilities = model.predict(x_test)

# Prompt user to enter ID to view image and resulting predicted class
i = int(input("Enter image ID (1, 10,000) to see that image (-1 to exit): "))
while i != -1:
    predicited_digit = np.argmax(digit_probabilities[i])

    title = "Predicted Digit: ", predicited_digit
    plt.title(title)
    plt.imshow(X_test[i], cmap=plt.cm.binary)
    plt.show()
    i = int(input("Enter image ID (1, 10,000) to see that image (-1 to exit): "))

# Print metrics
print("Test Accuracy", round(test_acc, 4))

# Plot confusion matrix 
# Note: confusion-matrix function is taken from the SKLEARN website: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.show()

digit_classes = np.argmax(digit_probabilities, axis=1)
output = np.argmax(y_test, axis=1)

confusion_mtx = confusion_matrix(output, digit_classes)
plot_confusion_matrix(confusion_mtx, classes = range(10))