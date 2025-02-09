# -*- coding: utf-8 -*-
"""Introduction_to_Deep_Learning_Project_3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uf4ZfZq6dDXtUg364bJ0K-SG9n2dIICp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import random
from tensorflow.keras.utils import load_img
warnings.filterwarnings('ignore')

"""**Downloading dataset:**"""

!wget https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip

"""**Unzip the dataset:**"""

!unzip kagglecatsanddogs_5340.zip

dset_pet_dir = "PetImages"

dset_pet = tf.keras.preprocessing.image_dataset_from_directory(dset_pet_dir)

"""**Listing directories:**


"""

!ls "PetImages"
#Or
dset_pet.class_names

"""**Exploratory Data Analysis:**"""

#Displaying image samples (label 0 is "cat" and label 1 is "dog")
plt.figure(figsize=(10, 10))
for images, labels in dset_pet.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

#Defining parameters for the loader:
batch_size = 64
img_height = 256
img_width = 256

# to display grid of images
plt.figure(figsize=(25,25))
temp = df[df['label']==1]['images']
start = random.randint(0, len(temp))
files = temp[start:start+25]

for index, file in enumerate(files):
    plt.subplot(5,5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title('Dogs')
    plt.axis('off')

# to display grid of images
plt.figure(figsize=(25,25))
temp = df[df['label']==0]['images']
start = random.randint(0, len(temp))
files = temp[start:start+25]

for index, file in enumerate(files):
    plt.subplot(5,5, index+1)
    img = load_img(file)
    img = np.array(img)
    plt.imshow(img)
    plt.title('Cats')
    plt.axis('off')

#Filtering out corrupted images
import os
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(dset_pet_dir, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()
        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)
print("Deleted %d images" % num_skipped)

#Data augmentation
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1)])

train_dset = tf.keras.preprocessing.image_dataset_from_directory(
    dset_pet_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

validation_dset = tf.keras.preprocessing.image_dataset_from_directory(
    dset_pet_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

for image_batch, labels_batch in train_dset:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

"""**Model Creation:**

Sequential CNN Model:
"""

model1 = tf.keras.models.Sequential([tf.keras.layers.Rescaling(1./255),
                                    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (256, 256, 3)),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Dropout(0.25),

                                    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    tf.keras.layers.Dense(2, activation = 'softmax')])

model1.compile(
  optimizer= tf.keras.optimizers.RMSprop(learning_rate=0.001),
  loss=tf.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy'])

history = model1.fit(train_dset, validation_data=validation_dset, epochs=100)

model1.summary()

model1.evaluate(validation_dset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

"""Pre trained Inceptionv3 model:"""

import os
import shutil
import glob

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras import layers
from tensorflow.keras import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import scipy
import numpy as np
import matplotlib.pyplot as plt

import PIL
import PIL.Image

weight = "/content/inception_v3_weights.h5"
print(type(weight))
print(" ")
print(weight)

inception_model = InceptionV3(input_shape = (256, 256, 3),
                                include_top = False,
                                weights = None)

inception_model.summary()

n_layers = 0
for layer in inception_model.layers:
    print(layer)
    n_layers += 1

print(" ")
print("Total layers of InceptionV3 Model are :", n_layers)

# This is the 'Input Layer'
inception_model.layers[0]

# This is the 1st 'Hidden Layer'
print("Layer Name ---> ", inception_model.layers[1])
print(" ")
inception_model.layers[1].get_weights()

weight_1 = inception_model.layers[1].get_weights()
print(len(weight_1))

#'weights' dimensions for the 1st Hiden Layer
weight_1_0 = np.array(weight_1[0])
print(weight_1_0.shape)

#freeze the layers of the model
for layer in inception_model.layers:
  layer.trainable = False
inception_model.summary()

from tensorflow.keras.optimizers import RMSprop

last_layer = inception_model.get_layer('mixed10')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)

#final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)

inception_model = Model(inception_model.input, x)

#data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator( rescale = 1.0/255. )

to_create = [
    'cats vs dogs',
    'cats vs dogs/training',
    'cats vs dogs/testing',
    'cats vs dogs/training/cats',
    'cats vs dogs/training/dogs',
    'cats vs dogs/testing/cats',
    'cats vs dogs/testing/dogs'
]

for directory in to_create:
    try:
        os.mkdir(directory)
        print(directory, 'created')
    except:
        print(directory, 'failed')

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
from shutil import copyfile

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    all_files = []

    for file_name in os.listdir(SOURCE):
        file_path = SOURCE + file_name

        if os.path.getsize(file_path):
            all_files.append(file_name)
        else:
            print('{} is zero length, so ignoring'.format(file_name))

    n_files = len(all_files)
    split_point = int(n_files * SPLIT_SIZE)

    shuffled = random.sample(all_files, n_files)

    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]

    for file_name in train_set:
        copyfile(SOURCE + file_name, TRAINING + file_name)

    for file_name in test_set:
        copyfile(SOURCE + file_name, TESTING + file_name)


CAT_SOURCE_DIR = "PetImages/Cat/"
TRAINING_CATS_DIR = "cats vs dogs/training/cats/"
TESTING_CATS_DIR = "cats vs dogs/testing/cats/"
DOG_SOURCE_DIR = "PetImages/Dog/"
TRAINING_DOGS_DIR = "cats vs dogs/training/dogs/"
TESTING_DOGS_DIR = "cats vs dogs/testing/dogs/"

split_size = .8
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)

print(len(os.listdir('cats vs dogs/training/cats/')))
print(len(os.listdir('cats vs dogs/training/dogs/')))
print(len(os.listdir('cats vs dogs/testing/cats/')))
print(len(os.listdir('cats vs dogs/testing/dogs/')))

train_dir = 'cats vs dogs/training/'

train_cats_dir = os.path.join(train_dir, 'cats') # Directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs') # Directory with our training dog pictures

print("train_cats_dir path is : ", train_cats_dir)
print(" ")
print("train_dogs_dir path is : ", train_dogs_dir)

valid_dir = 'cats vs dogs/testing/'
valid_cats_dir = os.path.join(valid_dir, 'cats') # Directory with our validation cat pictures
valid_dogs_dir = os.path.join(valid_dir, 'dogs')# Directory with our validation dog pictures

print(" ")
print("validation_cats_dir path is : ", valid_cats_dir)
print(" ")
print("validation_dogs_dir path is : ", valid_dogs_dir)

train_cat_fnames = os.listdir(train_cats_dir)
train_cat_fnames

train_dog_fnames = os.listdir(train_dogs_dir)
train_dog_fnames

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary',
                                                    target_size = (256, 256))

validation_generator =  test_datagen.flow_from_directory( valid_dir,
                                                          batch_size  = 20,
                                                          class_mode  = 'binary',
                                                          target_size = (256, 256))

# compile the model
inception_model.compile(optimizer=tf.optimizers.RMSprop(learning_rate = 0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss')

# train the model
history = inception_model.fit(train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 100,
            epochs = 100,
            validation_steps = 50,
            verbose = 2)

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss Graph')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy - Inception Model')
plt.legend(loc=0)
plt.figure()


plt.show()