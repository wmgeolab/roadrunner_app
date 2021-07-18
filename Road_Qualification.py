#!pip3 uninstall -y tensorflow
#!pip3 install tensorflow==2.0.0
#!pip3 uninstall -y tensorflow-gpu
#!pip3 install tensorflow-gpu==2.0.0

import datetime
import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow_hub as hub
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import itertools
import numpy as np
from numpy import expand_dims
from numpy import dstack
import matplotlib.pyplot as plt
import random
import csv
import math
from math import erf
import scipy.stats as stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
plt.rcParams["axes.grid"] = False


## create dataframes from csvs
#tv_fieldnames = ['id', 'filename','CQV','prob_label','hot_label','int_label']
#tv_df = pd.read_csv("tv_log.csv", usecols=tv_fieldnames)
#
#test_fieldnames = ['id','filename', 'int_label', 'CQV']
#test_df = pd.read_csv("test_log.csv", usecols=test_fieldnames)
#
## split into training and validation
#msk = np.random.rand(len(tv_df)) < 0.75
#train_df = tv_df[msk]
#val_df = tv_df[~msk]
#
## calculate class weights
#counts = train_df['int_label'].value_counts()
#low_num = counts[0]
#mid_num = counts[1]
#high_num = counts[2]
#classweights = {0: high_num/low_num, 1: high_num/mid_num, 2: 1.}
#classes = 3


# binary
fieldnames = ['id', 'filename', 'CQV', 'int_label']
binary_df = pd.read_csv("Low_MidHigh_Binary_log.csv", usecols=fieldnames)

msk = np.random.rand(len(binary_df)) < 0.75
tv_df = binary_df[msk]
test_df = binary_df[~msk]

msk = np.random.rand(len(tv_df)) < 0.80
train_df = tv_df[msk]
val_df = tv_df[~msk]

counts = train_df['int_label'].value_counts()
low_num = counts[0]
high_num = counts[1]
classweights = {0: high_num/low_num, 1: 1.}
classes = 2


img_dim = 200


# create training dataset

train_df['prob_label'] = ''
for index,row in train_df.iterrows():
  qv = row['CQV'] 
  sigma = 0.25
  label = []
  #bounds = [(-3,1),(1,2),(2,6)]
  bounds = [(-3,1),(2,6)]
  for l,u in bounds:
    # probability from Z=0 to lower bound
    double_prob = erf( (l-qv) / (sigma*math.sqrt(2)) )
    p_lower = double_prob/2
    # probability from Z=0 to upper bound
    double_prob = erf( (u-qv) / (sigma*math.sqrt(2)) )
    p_upper = double_prob/2
    prob = round(p_upper-p_lower,4)
    label.append(prob)
  train_df['prob_label'][index] = label

train_imgs=[]
train_labs = []
for index,row in train_df.iterrows():
  img = plt.imread(row['filename'])
  img = resize(img, (img_dim, img_dim))
  train_imgs.append(img)
  train_labs.append(row['prob_label'])
  #train_labs.append(row['CQV'])
train_imgs = np.stack(train_imgs)
train_imgs = train_imgs.reshape(-1, img_dim, img_dim, 3)
train_imgs = train_imgs.astype("float32")
train_labels = np.array(train_labs)


# create validation dataset
val_imgs=[]
val_labs = []
for index,row in val_df.iterrows():
  img = plt.imread(row['filename'])
  img = resize(img, (img_dim, img_dim))
  val_imgs.append(img)
  val_labs.append(row['int_label'])
  #val_labs.append(row['CQV'])
val_imgs = np.stack(val_imgs)
val_imgs = val_imgs.reshape(-1, img_dim, img_dim, 3)
val_imgs = val_imgs.astype("float32")
val_labels = to_categorical(val_labs, num_classes = classes)
#val_labels = np.array(val_labs)



# create test dataset
test_imgs=[]
test_labs = []
for index,row in test_df.iterrows():
  img = plt.imread(row['filename'])
  img = resize(img, (img_dim, img_dim))
  test_imgs.append(img)
  test_labs.append(row['int_label'])
  #test_labs.append(row['CQV'])
test_imgs = np.stack(test_imgs)
test_imgs = test_imgs.reshape(-1, img_dim, img_dim, 3)
test_imgs = test_imgs.astype("float32")
test_labels = to_categorical(test_labs, num_classes = classes)
#test_labels = np.array(test_labs)


# define function for finalizing Nigerian dataset structure
def finalize_dataset(imgs,labs,cls):
  images = np.stack(imgs)
  images = images.reshape(-1, img_dim, img_dim, 3)
  images = images.astype("float32")
  labels = to_categorical(labs, num_classes = cls)    
  return images, labels

# create Nigerian test dataset
NigeriaHQ_fp = os.listdir('HQ_African_Roads')
Nigeria_imgs = [] 
Nigeria_labs = []
for i in range(len(NigeriaHQ_fp)):
  img = plt.imread('HQ_African_Roads/' + NigeriaHQ_fp[i])
  img = resize(img,(img_dim,img_dim))
  Nigeria_imgs.append(img)
  Nigeria_labs.append(1)  
NigeriaLQ_fp = os.listdir('LQ_African_Roads')
for i in range(len(NigeriaLQ_fp)):
  img = plt.imread('LQ_African_Roads/' + NigeriaLQ_fp[i])
  img = resize(img,(img_dim,img_dim))
  Nigeria_imgs.append(img)
  Nigeria_labs.append(0)
Nigeria_images, Nigeria_labels = finalize_dataset(Nigeria_imgs,Nigeria_labs, classes)


# create train, val, and test sets for Nigeria images
#random_ints = random.sample(list(map(int, np.linspace(0,len(Nigeria_imgs)-1,len(Nigeria_imgs)).tolist())), round(len(Nigeria_imgs)))
#with open('Nigerian_rand_indeces.txt', 'w') as filehandle:
#  filehandle.writelines("%s\n" % i for i in random_ints)
   
with open('Nigerian_rand_indeces.txt') as f:
  txt_lines = f.readlines()
rand_ints = []
for line in txt_lines:
  rand_ints.append(int(line))
  
Nigeria_train_imgs = []
Nigeria_train_labs = []
Nigeria_val_imgs = []
Nigeria_val_labs = []
Nigeria_test_imgs = []
Nigeria_test_labs = []
a = 0
for i in rand_ints:
  if a < 600:
    Nigeria_train_imgs.append(Nigeria_imgs[i])
    Nigeria_train_labs.append(Nigeria_labs[i])
  if a >= 600 and a < 750:  
    Nigeria_val_imgs.append(Nigeria_imgs[i])
    Nigeria_val_labs.append(Nigeria_labs[i])
  if a >= 750:
    Nigeria_test_imgs.append(Nigeria_imgs[i])
    Nigeria_test_labs.append(Nigeria_labs[i])
  a += 1  
Nigeria_train_images, Nigeria_train_labels = finalize_dataset(Nigeria_train_imgs,Nigeria_train_labs, classes)
Nigeria_val_images, Nigeria_val_labels = finalize_dataset(Nigeria_val_imgs,Nigeria_val_labs, classes)
Nigeria_test_images, Nigeria_test_labels = finalize_dataset(Nigeria_test_imgs,Nigeria_test_labs, classes)

classweights = {0: Nigeria_train_labs.count(1)/Nigeria_train_labs.count(0), 1: 1.}


# choose base model
"""
base_model = tf.keras.applications.Xception(
  weights="imagenet",
  input_shape=(img_dim, img_dim, 3),
  include_top=False)
bm_name = 'Xception'
"""

"""
base_model = tf.keras.applications.ResNet50V2(
  weights="imagenet",
  input_shape=(img_dim, img_dim, 3),
  include_top=False)
bm_name = 'ResNet50'
"""

"""
base_model = tf.keras.applications.ResNet152V2(
  weights="imagenet",
  input_shape=(img_dim, img_dim, 3),
  include_top=False)
bm_name = 'ResNet152' 
"""

"""
base_model = tf.keras.applications.InceptionV3(
  weights="imagenet",
  input_shape=(img_dim, img_dim, 3),
  include_top=False)
bm_name = 'Inception'  
"""

"""
base_model = tf.keras.applications.VGG16(
  weights="imagenet",
  input_shape=(img_dim, img_dim, 3),
  include_top=False)
bm_name = 'VGG'
"""


"""
base_model = tf.keras.applications.DenseNet201(
  weights="imagenet",
  input_shape=(img_dim, img_dim, 3),
  include_top=False)
bm_name = 'DenseNet'  
"""


base_model = tf.keras.applications.InceptionResNetV2(
  weights="imagenet",
  input_shape=(img_dim, img_dim, 3),
  include_top=False)
bm_name = 'InceptionResNet'


"""
base_model = hub.KerasLayer("https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1")
bm_name = 'BigEarthResNet'
"""

# assemble network
model = Sequential()
model.add(tf.keras.Input((img_dim,img_dim,3)))
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(classes, activation = 'softmax'))
# freeze weights in base model
base_model.trainable = False


# compile model with frozen base model weights
model.compile(optimizer = tf.keras.optimizers.Adam(0.0001),
              loss = tf.keras.losses.CategoricalCrossentropy(),
              #loss = tf.keras.losses.MeanSquaredError(),
              metrics = ['accuracy'])
              #metrics=['MeanSquaredError'])
model.summary()


# define early stopping for first round and batch size
es1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)
bs = 16

# train model with frozen base model weights
model.fit(x = train_imgs, 
          y = train_labels,
          validation_data = (val_imgs, val_labels),
          batch_size = bs,
          epochs = 2,
          verbose = 2,
          callbacks = [es1],
          class_weight = classweights)



# make layers in base model trainable
base_model.trainable = True

# Unfreeze only some base model layers
#for layer in base_model.layers[0:]:
#  layer.trainable = False 
# VGG  
#for layer in base_model.layers[15:]:
#  layer.trainable = True
# DenseNet
#for layer in base_model.layers[698:]:
#  layer.trainable = True
#i=0
#for layer in base_model.layers:
#  print("{}: {}".format(layer, layer.trainable))
#  i+=1

# compile complete model
model.compile(optimizer = tf.keras.optimizers.Adam(0.00001),
              loss = tf.keras.losses.CategoricalCrossentropy(),
              #loss = tf.keras.losses.MeanSquaredError(),
              metrics = ['accuracy'])
              #metrics=['MeanSquaredError'])
#model = tf.keras.models.load_model('Saved_Binary_Class_Models/InceptionResNet_04122021-210053_0.9447')
model.summary()


# define early stopping for second round and model checkpoint
es2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
#mc = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_MeanSquaredError', mode='min', verbose=1, save_best_only=True)


# train complete model
train = model.fit(x = train_imgs, 
                  y = train_labels,
                  validation_data = (val_imgs, val_labels),
                  batch_size = bs,
                  epochs = 100,
                  verbose = 2,
                  callbacks = [es2, mc],
                  class_weight = classweights)




# select best-trained complete model
final_model = tf.keras.models.load_model('best_model.h5')

#final_model = tf.keras.models.load_model('Saved_Africa_Fine_Tune/InceptionResNet_Low_High_04132021-144425_0.94')
#final_model.summary()


# test network and show final accuracy and loss for classification
test_eval = final_model.evaluate(x=Nigeria_test_images, y=Nigeria_test_labels, verbose=0)
print('Test accuracy:', test_eval[1])
print('Test loss:', test_eval[0])


# generate predictions for classification
predicted_probs = final_model.predict(Nigeria_test_images)
predictions = np.argmax(predicted_probs, axis=1)

# write predictions to test_results.csv,
p=0
with open('predictions.csv', 'w', newline='') as test_results:
  results_fieldnames = ['id', 'prediction']
  results_writer = csv.DictWriter(test_results, fieldnames=results_fieldnames)
  results_writer.writeheader()
  with open('/var/lib/cdsw/share/test_log.csv', 'r') as test_log:
    test_log_reader = csv.reader(test_log)
    next(test_log_reader)
    for row in test_log_reader:
      idnum = row[0]
      results_writer.writerow({'id': idnum, 'prediction': predictions[p]})
      p += 1
      if p == len(predictions):
        break

"""
# calculate error measures for CQV estimate
predicted_CQVs = final_model.predict(test_imgs)
MSE = mean_squared_error(test_labs, predicted_CQVs)
MAE = mean_absolute_error(test_labs, predicted_CQVs)
print('Mean Squared Error: %.3f' % MSE)
print('Mean Absolute Error: %.3f' % MAE)
"""

# save model
save_path = "Saved_Low_MidHigh_Binary_Models/" + bm_name + '_' + datetime.datetime.now().strftime("%m%d%Y-%H%M%S") + '_' + str(round(test_eval[1],4))
#save_path = "/home/cdsw/Saved_CQV_Models/" + bm_name + '_' + datetime.datetime.now().strftime("%m%d%Y-%H%M%S") + '_' + str(round(MSE,4))
shutil.copyfile('best_model.h5', save_path)

# Save training history
np.save('training_history.npy', train.history)











##############
#  Ensemble  #
##############


# sort by CQV and take all low quality roads and twice as many high quality
tv_df = pd.read_csv("tv_log.csv", usecols=train_fieldnames)
tv_df = tv_df.sort_values('CQV')
tv_df = pd.concat([tv_df[:497], tv_df[2700:3500], tv_df[51000:52821]], axis=0)
tv_df.groupby('int_label').count()
tv_imgs=[]
tv_labs = []
for index,row in tv_df.iterrows():
  img = plt.imread(row['filename'])
  img = resize(img, (img_dim, img_dim))
  tv_imgs.append(img)
  #tv_labs.append(row['int_label'])
  tv_labs.append(row['CQV'])
tv_imgs = np.stack(tv_imgs)
tv_imgs = tv_imgs.reshape(-1, img_dim, img_dim, 3)
tv_imgs = tv_imgs.astype("float32")


# load models
#files = os.listdir(r"Saved_Class_Models/Ensemble")
files = os.listdir(r"Saved_CQV_Models/Ensemble")
members = []
for file in files:
  #model = tf.keras.models.load_model('Saved_Class_Models/Ensemble/'+file)
  model = tf.keras.models.load_model('Saved_CQV_Models/Ensemble/'+file)
  members.append(model)
  print('loaded %s' % file)
  
  
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX
  

# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members, inputX, inputy):
  # create dataset using ensemble
  stackedX = stacked_dataset(members, inputX)
  # fit standalone model
  #model = LogisticRegression()
  model = LinearRegression()
  model.fit(stackedX, inputy)
  return model


# fit stacked model using the ensemble
stacked_model = fit_stacked_model(members, tv_imgs, tv_labs)


# make a prediction with the stacked model
def stacked_prediction(members, model, inputX):
	# create dataset using ensemble
	stackedX = stacked_dataset(members, inputX)
	# make a prediction
	yhat = model.predict(stackedX)
	return yhat


# evaluate model on test set
yhat = stacked_prediction(members, stacked_model, test_imgs)
#acc = accuracy_score(test_labs, yhat)
MSE = mean_squared_error(test_labs, yhat)
MAE = mean_absolute_error(test_labs, yhat)
#print('Stacked Ensemble Test Accuracy: %.3f' % acc)
print('Stacked Ensemble Mean Squared Error: %.3f' % MSE)
print('Stacked Ensemble Mean Absolute Error: %.3f' % MAE)



# write predictions to test_results.csv,
p=0
with open('predictions.csv', 'w', newline='') as test_results:
  results_fieldnames = ['id', 'prediction']
  results_writer = csv.DictWriter(test_results, fieldnames=results_fieldnames)
  results_writer.writeheader()
  with open('/var/lib/cdsw/share/test_log.csv', 'r') as test_log:
    test_log_reader = csv.reader(test_log)
    next(test_log_reader)
    for row in test_log_reader:
      idnum = row[0]
      results_writer.writerow({'id': idnum, 'prediction': yhat[p]})
      p += 1
      if p == len(yhat):
        break







      
# display results table
target_names = ["Unpaved Roads", "Paved Roads"]
print(classification_report(Nigeria_test_labs, predictions, target_names=target_names))  
  

# define confusion matrix (left truth, top predictions)
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    #pretty print for confusion matrixes
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            
            try:
                cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            except IndexError:
                cell = "     0"
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")    
        print()


# show confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Nigeria_test_labs, predictions)
print_cm(cm, target_names, hide_zeroes = False)


# show random correctly classified images
correct = np.where(yhat==test_labs)[0]
plt.figure(figsize=(10, 10))
for i in range(9):
  rand_num = random.randint(0, len(correct)-1)
  rand_image = test_imgs[correct[rand_num]]
  label = test_labs[correct[rand_num]]
  if label == 0:
    lab = 'low'
  elif label == 1:
    lab = 'mid'
  else:
    lab = 'high'
  plt.subplot(3, 3, i + 1)
  plt.imshow(rand_image)
  plt.title(lab)
  plt.axis("off")

  
# show random incorrectly classified images
incorrect = np.where(yhat!=test_labs)[0]
plt.figure(figsize=(10, 10))
for i in range(9):
  rand_num = random.randint(0, len(incorrect)-1)
  rand_image = test_imgs[incorrect[rand_num]]
  label = test_labs[incorrect[rand_num]]
  if label == 0:
    lab = 'low'
  elif label == 1:
    lab = 'mid'
  else:
    lab = 'high'
  prediction = yhat[incorrect[rand_num]]
  if prediction == 0:
    pred = 'low'
  elif prediction == 1:
    pred = 'mid'
  else:
    pred = 'high'
  plt.subplot(3, 3, i + 1)
  plt.imshow(rand_image)
  plt.title('  ' + 'pred: ' + pred + ',  label: ' + lab + '  ')
  plt.axis("off")

  


"""
# show random training images
labs = []
for lab in train_labs[0:9]:
  lab = np.argmax(lab, axis=0)
  labs.append(lab)
plt.figure(figsize=(10, 10))
for i in range(9):
  plt.subplot(3, 3, i + 1)
  plt.imshow(train_imgs[i])
  plt.title(int(labs[i]))
  plt.axis("off")
   

# show random validation images
plt.figure(figsize=(10, 10))
for i in range(9):
  plt.subplot(3, 3, i + 1)
  plt.imshow(val_imgs[i])
  plt.title(int(val_labs[i]))
  plt.axis("off")
  
  
# show random test images
plt.figure(figsize=(10, 10))
for i in range(9):
  plt.subplot(3, 3, i + 1)
  plt.imshow(test_imgs[i])
  plt.title(test_labs[i])
  plt.axis("off")  

  
# show random Nigerian images
plt.figure(figsize=(10, 10))
for i in range(9):
  plt.subplot(3, 3, i + 1)
  plt.imshow(Nigeria_train_images[i+37])
  plt.title(Nigeria_train_labs[i+37])
  plt.axis("off")    

"""





"""
    
############################
#  Visualize Layer Output  #
############################


base_model.layers

# summarize feature map shapes
for i in range(len(base_model.layers)):
	layer = base_model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)


# redefine model to output right after specified layers
ixs = [2, 9, 17]
outputs = [base_model.layers[i+1].output for i in ixs]
vis_model = Model(inputs=base_model.inputs, outputs=outputs)
vis_model.summary()

img_num = 0
img = test_imgs[img_num]
plt.imshow(img)


# get feature maps for layers
img = expand_dims(img, axis=0)
feature_maps = vis_model.predict(img)


# plot the output from each block
square = 5
for fmap in feature_maps:
  ix = 1
  for _ in range(square):
    for _ in range(square):
      ax = plt.subplot(square, square, ix)
      ax.set_xticks([])
      ax.set_yticks([])
      plt.imshow(fmap[0, :, :, ix-1], cmap='gray')
      ix += 1
  plt.show()

""" 


 
  


"""   
# Find optimal hyperparameters from grid search

# Function to create model, required for KerasClassifier
def create_model(learn_rate):
  # define model
  base_model.trainable = True
  inputs = tf.keras.Input(shape=(img_dim, img_dim, 3))
  x = base_model(inputs, training=False)
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(64)(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
  model = tf.keras.Model(inputs, outputs)
  model.compile(optimizer = tf.keras.optimizers.Adam(learn_rate),
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ['accuracy'])
  return model
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# define data
X = train_ds[0][0]
Y = train_labels
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [32, 64, 128]
learn_rate = [1e-6,1e-7, 1e-8]
param_grid = dict(batch_size=batch_size, learn_rate=learn_rate)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))  
"""


  
  
  
  
  

"""
################################################
#  Analyze data and create labels and dataset  #
################################################


log_filepath = "final_log.csv"
fieldnames = ['id', 'line', 'distance', 'iriXp', 'iriYp', 'iriZp', 'iriX', 'iriY', 'iriZ', 'filename', 'time', 'phoneID']
df = pd.read_csv(log_filepath, usecols=fieldnames)

# convert segment filenames to image paths
satellite_image_path_prefix = "/var/lib/cdsw/share/Ethan_pngs/"
satellite_image_path_suffix = "_16.png"
for index, row in df.iterrows():   
  image_path = satellite_image_path_prefix + df["filename"][index] + satellite_image_path_suffix   
  df["filename"][index] = image_path


# create CQV and labels
df['CQV'] = ''
df['prob_label'] = ''
df['hot_label'] = ''
df['int_label'] = ''
all_qv = []
all_hot = []
all_pos = []
for index,row in df.iterrows():
  
  # Charles - VT
  if row["phoneID"] == 'f6f8dc81da1eb34f':
    rough = math.sqrt((row['iriXp'])**2 + (row['iriYp'])**2 + (row['iriZp'])**2)
    worse = 0.8
    better = 0.4
  
  # Vivi - VT
  elif row["phoneID"] == '1ad1d9324d8dd616':
    rough = math.sqrt((row['iriX'])**2 + (row['iriY'])**2 + (row['iriZ'])**2)
    worse = 0.16
    better = 0.07
  
  # Minh - VT
  elif row["phoneID"] == 'a162e1304c61808e': 
    rough = row['iriZ']
    worse = 0.6
    better = 0.375   
  
  # Taima - Richmond
  elif row["phoneID"] == '26f7270e39098e76':
    rough = math.sqrt((row['iriXp'])**2 + (row['iriYp'])**2 + (row['iriZp'])**2)
    worse = 4
    better = 1.8
  
  # Calvin - GMU
  elif row["phoneID"] == 'a831546857f534ea':
    rough = row['iriZ']
    worse = 0.2
    better = 0.075 
  
  # Mathhew - GMU
  elif row["phoneID"] == '0721eb613440430a':
    rough = row['iriZ'] 
    if (row['time'] >= 1599883200000 and row['time'] < 1599969600000) | (row['time'] >= 1601092800000 and row['time'] < 1601179200000) | (row['time'] >= 1603684800000 and row['time'] < 1603771200000):
      worse = 0.24
      better = 0.12
    elif (row['time'] >= 1600401600000 and row['time'] < 1600488000000) | (row['time'] >= 1605070800000 and row['time'] < 1605157200000):
      worse = 0.4
      better = 0.157
    elif (row['time'] >= 1601179200000 and row['time'] < 1601265600000):
      worse = 0.25
      better = 0.125
    else:
      raise ValueError
  
  # Allan - CNU
  elif row["phoneID"] == 'acc836a88b20fd37':
    rough = math.sqrt((row['iriX'])**2 + (row['iriY'])**2 + (row['iriZ'])**2)
    worse = 0.25
    better = 0.07 
    
  # Jacob - W&M
  elif row["phoneID"] == 'e3c058a7bc07c878':
    rough = row['iriZ']
    worse = 0.141
    better = 0.022   
  
  # Grace - VCU
  elif row["phoneID"] == '18151f255f1d95d6':
    rough = math.sqrt((row['iriXp'])**2 + (row['iriYp'])**2 + (row['iriZp'])**2)
    worse = 2
    better = 0.9 
  
  # Ethan
  else:
    try:
      math.isnan(row["phoneID"])
      rough = row['iriZ']
      worse = 0.2494 
      better = 0.1247
    except:
      print(row['phoneID'])
      print(index)
      raise ValueError
  
  if rough >= worse:
    qv = 3 - (2*(rough/worse))
    if qv < 0:
      qv = 0
    all_qv.append(qv) 
  elif rough < worse and rough >= better:
    qv = 2 - (rough-better)/(worse-better)
    all_qv.append(qv) 
  elif rough < better:
    qv = 3 - (rough/better)
    all_qv.append(qv) 
  else:
    raise ValueError    
    
  sigma = 0.25
  label = []
  bounds = [(-3,1),(1,2),(2,6)]
  for l,u in bounds:
    # probability from Z=0 to lower bound
    double_prob = erf( (l-qv) / (sigma*math.sqrt(2)) )
    p_lower = double_prob/2
    # probability from Z=0 to upper bound
    double_prob = erf( (u-qv) / (sigma*math.sqrt(2)) )
    p_upper = double_prob/2
    prob = round(p_upper-p_lower,4)
    label.append(prob)
  df['prob_label'][index] = label
  pos = np.argmax(df['prob_label'][index])
  all_pos.append(str(pos))  
  hot = to_categorical(pos, num_classes = 3)
  all_hot.append(hot)
df['CQV'] = all_qv
df["hot_label"] = all_hot
df['int_label']= all_pos
# shuffle df
df = df.sample(frac=1).reset_index(drop=True)
df.groupby('int_label').count()

# create dataframes of each class
low_df = df.loc[df['int_label'].isin(['0'])].groupby('int_label').head(712)
mid_df = df.loc[df['int_label'].isin(['1'])].groupby('int_label').head(5417)
high_df = df.loc[df['int_label'].isin(['2'])].groupby('int_label').head(47557)

# split each classed dataframe into training and testing dfs
msk_low = np.random.rand(len(low_df)) < 0.7
train_low_df = low_df[msk_low]
test_low_df = low_df[~msk_low]

msk_mid = np.random.rand(len(mid_df)) < (1-(len(test_low_df)/len(mid_df)))
train_mid_df = mid_df[msk_mid]
test_mid_df = mid_df[~msk_mid]

msk_high = np.random.rand(len(high_df)) < (1-(2*len(test_low_df)/len(high_df)))
train_high_df = high_df[msk_high]
test_high_df = high_df[~msk_high]

# merge each of the training and testing dfs
train_frames = [train_low_df, train_mid_df, train_high_df]
test_frames = [test_low_df, test_mid_df, test_high_df]
train_df = pd.concat(train_frames)
test_df = pd.concat(test_frames)
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

test_df.groupby('int_label').count()

# select columns and save dfs as csvs
sel_train_df = train_df[['id','filename','CQV','prob_label','hot_label','int_label']].copy()
sel_test_df = test_df[['id','filename', 'CQV', 'int_label',]].copy()
sel_train_df.to_csv("train_log.csv", index=False)
sel_test_df.to_csv("test_log.csv", index=False)

## make special test log
#special_df = test_df[['id','int_label']].copy()
#special_df.to_csv("special_test_log.csv", index=False)

"""
  

"""
# add CQV to test log
sp_test_df['CQV']=''
allcqv = []
for tindex,row in sp_test_df.iterrows():
  iden = row['id']
  logindex = df[df['id'] == iden].index[0]
  CQV = df["CQV"][logindex]
  allcqv.append(CQV)
sp_test_df['CQV'] = allcqv
sp_test_df.to_csv("special_test_log2.csv", index=False)  
"""  


"""
# replace filename path in df and create new log
for index, row in binary_df.iterrows():   
  string = binary_df["filename"][index]
  image_path = string.replace('Ethan_pngs','/var/lib/cdsw/share/Ethan_pngs')
  binary_df["filename"][index] = image_path

""" 

"""
# drop data with no images
nooverlap = [121167,209535,258156,374508,210829,397225,97737,441358,589168]
for index, row in test_df.iterrows():   
  ID = test_df["id"][index]
  if ID in nooverlap:
    test_df = test_df.drop(index=index, axis=0)
test_df.to_csv("test_log.csv", index=False)    
"""    


