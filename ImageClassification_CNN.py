
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
# import cv2


# In[2]:


import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

# TRAIN_IMG_DIR = "/data/examples/may_the_4_be_with_u/where_am_i/train/"
# TEST_IMG_DIR = "/data/examples/may_the_4_be_with_u/where_am_i/testset/"
TRAIN_IMG_DIR = "./train/"
TEST_IMG_DIR = "./testset/"
# VALID_IMG_DIR ="./validation/"

train_samples = 2985 #total 2985 images in train_img_dir belonging to 15 classes
test_samples = 1500 #1500 images in test_img_dir
num_classes = 15 #target labels(ground truth), total 15 classes

# image shapes
img_width = 128
img_height = 128
channels = 3
input_shape = (img_width, img_height, channels)

batch_size = 16
epochs = 100

## Build model: Conv layer + MaxPooling layer + fully-connected NN layer
#Conv layer + MaxPooling layer
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape)) #input_shape argument must be assigned in first layer!
model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
# model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))

#fully-connected NN layer
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation="softmax"))
# !change Activation from keras to tf.nn.softmax, because TF version too old on Server!
model.add(Dense(num_classes))
import tensorflow as tf
model.add(Activation(tf.nn.softmax))

# opt_adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False)
model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

print (model.summary())


# In[3]:


## Using Keras ImageDataGenerator to load images batch and do data augmentation on the fly.
#!validation_split argument not support in Keras 2.1.3(server version)!
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        validation_split = 0.33 
)

valid_datagen = ImageDataGenerator(
        rescale = 1./255,
#         rotation_range = 20,
#         width_shift_range = 0.2,
#         height_shift_range = 0.2,
#         shear_range = 0.2,
#         zoom_range = 0.2,
#         horizontal_flip = True,
        validation_split = 0.33
)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        directory = TRAIN_IMG_DIR,
        target_size = (img_width, img_height),
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True,
        seed = 33,
        subset = "training"
)

validation_generator = valid_datagen.flow_from_directory(
        directory = TRAIN_IMG_DIR,
        target_size = (img_width, img_height),
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True,
        seed = 33,
        subset = "validation"
)

test_generator = test_datagen.flow_from_directory(
        directory = TEST_IMG_DIR,
        target_size = (img_width, img_height),
        color_mode = "rgb",
        batch_size = 1,
        class_mode = None,
        shuffle = False,
)

## Amounts of individual set: training, validation, test
# train_generator.n #amounts of training set
# validation_generator.n #amounts of validation set
# test_generator.n #amounts of test set

## Labels from Keras data generator
# print (train_generator.class_indices)
# print (validation_generator.class_indices)

## Image shape check
print (train_generator.image_shape)
print (validation_generator.image_shape)
print (test_generator.image_shape)


# In[4]:


## Fitting/Training the model
# steps_per_epoch = train_samples // batch_size
steps_per_epoch = train_generator.n // batch_size
validation_steps = validation_generator.n // batch_size

# Callbacks setting
filepath = "./checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5"
EarlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
Checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
Callback_list = [EarlyStopping, Checkpoint]

history = model.fit_generator(
                generator = train_generator,
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                callbacks = Callback_list,
                validation_data = validation_generator,
                validation_steps = validation_steps,
#                 validation_data = None,
#                 validation_steps = None,
                shuffle = True
)

## Evaluate the model
# model.evaluate_generator(generator = )

## Predict the test set, then we'll get a probability nparray
test_generator.reset()
pred_probability = model.predict_generator(test_generator, verbose=1)

## Convert the probability nparray to pandas dataframe to see its structure
# df_pred = pd.DataFrame(pred_probability)
# display(df_pred)


# In[5]:


## Get the predicted class indices from model prediction result
predicted_class_indices = np.argmax(pred_probability, axis=1)
#default labels from Keras data generator
keras_labels = (train_generator.class_indices)
#get the names of class labels
keras_labels_swap = dict((value, key) for key, value in keras_labels.items())
class_name = [keras_labels_swap[idx] for idx in predicted_class_indices]

## Reading pre-defined labels from mapping.txt, and store it to a dictionary
mapping = {}
with open("./mapping.txt") as f:
    for line in f:
        (key, val) = line.split(sep=",")
        mapping[str(key)] = int(val)

## Because predicted_class_indices comes from Keras (data generator) default labels,
## this is not our pre-defined labels (from mapping.txt).
## I use pandas.Series.map(arg=Dict) to remap predicted_class_indices to pre-defined labels.
ps = pd.Series(data = class_name)
class_predictions = ps.map(mapping)

## Save the results to a csv file
#first, get filenames of all test images
files = test_generator.filenames #this output will include the directory name!
#use regular expression to retrieve exact filename of test images
import re
filenames = []
for num in range(len(files)):
    lst = re.findall("testimg/([a-zA-Z0-9]+).jpg", files[num])
    for idx, value in enumerate(lst):
        filenames.append(value)

#save the results to a csv file
results = pd.DataFrame({"id" : filenames,
                        "class_name" : class_name,
                        "class" : class_predictions})
results.to_csv("results.csv", index=False)

submission = pd.DataFrame({"id" : filenames,
                           "class" : class_predictions})
submission.to_csv("submission.csv", index=False)


# In[6]:


get_ipython().system('jupyter nbconvert --to script ImageClassification_CNN.ipynb')

