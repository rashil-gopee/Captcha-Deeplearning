import cv2
import pickle
import os.path
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils.vis_utils import plot_model
import tensorflow as tf

LETTER_IMAGES_FOLDER = "cleaned_data_set"
MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"


# initialize the data and labels
data = []
labels = []

# loop over the input images

with tf.device("/gpu:0"):
    for image_file in paths.list_images(LETTER_IMAGES_FOLDER):
        # Load the image and convert it to grayscale
        image = cv2.imread(image_file, 0)

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the letter so it fits in a 20x20 pixel box

        # Add a third channel dimension to the image to make Keras happy
        image = np.expand_dims(image, axis=2)

        # Grab the name of the letter based on the folder it was in
        label = image_file.split(os.path.sep)[-2]

        # Add the letter image and it's label to our training data
        data.append(image)
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1] (this improves training)
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # Split the training data into separate train and test sets
    (X_train, X_test, Y_train, Y_test) = train_test_split(
        data, labels, test_size=0.20, random_state=0)

    # Convert the labels (letters) into one-hot encodings that Keras can work with
    lb = LabelBinarizer().fit(Y_train)
    Y_train = lb.transform(Y_train)
    Y_test = lb.transform(Y_test)

    # Save the mapping from labels to one-hot encodings.
    # We'll need this later when we use the model to decode what it's predictions mean
    with open(MODEL_LABELS_FILENAME, "wb") as f:
        pickle.dump(lb, f)

    # Build the neural network!
    model = Sequential()

    # First convolutional layer with max pooling
    model.add(Conv2D(32, (5, 5), padding="same",
                     input_shape=(80, 25, 1), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))
    # Second convolutional layer with max pooling
    model.add(Conv2D(64, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # Second convolutional layer with max pooling
    model.add(Conv2D(128, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))
    # Second convolutional layer with max pooling
    model.add(Conv2D(256, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.3))

    # Hidden layer with 500 nodes
    model.add(Flatten())
    model.add(Dense(1000, activation="relu"))

    # Output layer with 62 nodes (one for each possible letter/number we predict)
    model.add(Dense(62, activation="softmax"))

    # Ask Keras to build the TensorFlow model behind the scenes
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])

    # Train the neural network
    history = model.fit(X_train, Y_train, validation_data=(
        X_test, Y_test), batch_size=32, epochs=150, verbose=1)

print(history.history.keys())
plot_model(model, to_file='model_plot.png',
           show_shapes=True, show_layer_names=True)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Save the trained model to disk
model.save(MODEL_FILENAME)
