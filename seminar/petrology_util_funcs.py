'''
This Petrology utility functions are produced
by T. Ansah-Narh
Friday 16 April 2021
'''

# ++++++++++++++++++++++++++++++
# << Import Python libraries >>
# ++++++++++++++++++++++++++++++
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
# from random import sample
import glob
import shutil
from collections import Counter
import cv2
import itertools
import random
from subprocess import check_output
# <<keras modules>>
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
# from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
import visualkeras

from sklearn.model_selection import train_test_split


def split_stratified_into_train_val_test(df_input, frac_train=0.6, frac_val=0.15, frac_test=0.25, random_state=None, num=None):
	
	'''
	Splits a set of data into three subsets (train, validation, and test)
	following fractional ratios provided by the user, where each subset is
	stratified by the values in a specific column (that is, each subset has
	the same relative frequency of the values in the column). It performs this
	splitting by running train_test_split() twice.

	Parameters
	----------
	df_input : list of data
		Input dataset to split.
	frac_train : float
	frac_val   : float
	frac_test  : float
		The ratios with which the dataframe will be split into train, val, and
		test data. The values should be expressed as float fractions and should
		sum to 1.0.
	random_state : int, None, or RandomStateInstance
		Value to be passed to train_test_split().

	Returns
	-------
	train, val, test :
		list of data containing the three splits.
	'''

	if frac_train + frac_val + frac_test != 1.0:
		raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
						 (frac_train, frac_val, frac_test))


	X = os.listdir(df_input)[:num] # Contains all columns.
	y = list(range(len(X)))
	# Split original dataframe into train and temp dataframes.
	train, temp, y_train, y_temp = train_test_split(X, y, stratify=None, test_size=(1.0 - frac_train), random_state=random_state)

	# Split the temp data into val and test sets.
	relative_frac_test = frac_test / (frac_val + frac_test)
	val, test, y_val, y_test = train_test_split(temp, y_temp, stratify=None, test_size=relative_frac_test, random_state=random_state)
	# print(len(df_train) + len(df_val) + len(df_test))

	assert len(X) == len(train) + len(val) + len(test)

	return train, test, val


def plot_image(img_arr, fsize=(20,20), DPI=100):
	fig,axes= plt.subplots(1,10, figsize=fsize, dpi=DPI)
	axes= axes.flatten()
	for img,ax in (img_arr, axes):
		ax.imshow(img)
		ax.axis('off')
	plt.tight_layout()
	plt.show()

	
def percentage_value(pct, allvals):

	absolute = int(pct/100.*np.sum(allvals))
	
	return "{:.1f}\%\n({:d})".format(pct, absolute)


def plot_piechart(path, title, fsize=(8,8), DPI=100):

	classes = []
	for filename in glob.iglob(os.path.join(path, "**","*.???")):
		classes.append(os.path.split(os.path.split(filename)[0])[-1])

	classes_cnt = Counter(classes)
	values = list(classes_cnt.values())
	labels = list(classes_cnt.keys())

	plt.figure(figsize=fsize, dpi=DPI)
	plt.pie(values, 
			labels=labels, 
			autopct=lambda pct: percentage_value(pct, values), 
			shadow=True, 
			startangle=140)

	plt.title(title)
	plt.show()


def get_model(input_shape, num_classes=3, mname = 'VGG16', opt= SGD(lr=0.0001, momentum=0.9), view_summary=False, display_architecture=False):
	
	if (mname == 'VGG16' or mname == 'VGG19'): 
		if mname == 'VGG16':
			model = VGG16(weights = "imagenet", include_top=False, input_shape = (input_shape))
		else:
			model = VGG19(weights = "imagenet", include_top=False, input_shape = (input_shape))	 
		

		# Freeze the layers which you don't want to train. Freezing the first 5 layers.
		for layer in model.layers[:4]:
			layer.trainable = False			

		# Adding custom Layers
		x = model.output
		x = Flatten()(x)
		x = Dense(units=1024, activation="relu")(x)
		x = Dropout(0.5)(x)
		x = Dense(units=1024, activation="relu")(x)

	elif mname == 'InceptionV3': 
		model = InceptionV3(weights = "imagenet", include_top=False, input_shape = (input_shape))
		model.trainable = False
		x = base_model.output
		x = keras.layers.GlobalAveragePooling2D()(x)
		# let's add a fully-connected layer
		x = Dropout(0.5)(x)
	

	predictions = Dense(units=num_classes, activation="softmax")(x)

	# Creating the final model
	model_final = Model(inputs = model.input, outputs = predictions)

	# Compile the model
	# opt = RMSprop(lr=0.0001, decay=1e-6)
	# opt = SGD(lr=0.0001, momentum=0.9)
	# loss = categorical_crossentropy for multiclassification binary_crossentropy

	model_final.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=["accuracy"])

	if  view_summary is not False:
		print(model_final.summary())

	if  display_architecture is not False:
		visualkeras.layered_view(model=model_final, spacing=50).show()


	return model_final


def train_model(model, train_generator, validation_generator, batchSize, nepochs=3, weight_prefix='model', add_earlyStopping=True):
	'''
	Trains the model on data generated batch by batch.

	'''

	filepath=weight_prefix+"_bestweights.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	
	if add_earlyStopping is  True:
		early = EarlyStopping(monitor="val_loss", mode="min", patience=1)
		callbacks_list = [checkpoint, early]

	else:
		callbacks_list = [checkpoint]

	# Calculate steps_per_epoch and validation_steps For fit_generator
	stepsPerEpoch = train_generator.samples // batchSize
	print("stepsPerEpoch: ", stepsPerEpoch)

	validationSteps = validation_generator.samples // batchSize
	print("validationSteps: ", validationSteps)

	# Train
	train_generator.reset()
	validation_generator.reset()
	# Fit the model
	hist = model.fit(train_generator, validation_data = validation_generator, epochs = nepochs, steps_per_epoch = stepsPerEpoch, \
    							  validation_steps= validationSteps, callbacks=callbacks_list, verbose=1)

	return hist



def Vis_ConvLayers(model, img_path, ext_layers=[1,12], images_per_row = 16):
	'''
	This function produces feature maps from selected
	convolutional layers'''

	# Extract image from the path
	s_x = model.input.get_shape()[1]
	s_y = model.input.get_shape()[2]
	img = load_img(img_path, target_size=(s_x, s_y), interpolation='bicubic')
	img_tensor = img_to_array(img)
	img_tensor = np.expand_dims(img_tensor, axis=0)
	img_tensor /= 255.

	# Extracts the outputs of the selected layers 
	layer_outputs = [layer.output for layer in model.layers[ext_layers[0]:ext_layers[-1]]]
	
	# Creates a model that will return the outputs, given the model input
	activation_model = Model(inputs=model.input, outputs=layer_outputs)

	activations = activation_model.predict(img_tensor) 

	# Get layer names
	layer_names = []
	for layer in model.layers[ext_layers[0]:ext_layers[-1]]:
		layer_names.append(layer.name) 


	# Displays the feature maps
	for layer_name, layer_activation in zip(layer_names, activations):

		n_features = layer_activation.shape[-1] # Number of features in the feature map
		size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
		n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
		display_grid = np.zeros((size * n_cols, images_per_row * size))
		for col in range(n_cols): # Tiles each filter into a big horizontal grid
			for row in range(images_per_row):
				channel_image = layer_activation[0, :, :, col * images_per_row + row]
				channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
				channel_image /= channel_image.std()
				channel_image *= 64
				channel_image += 128
				channel_image = np.clip(channel_image, 0, 255).astype('uint8')
				display_grid[col * size : (col + 1) * size, # Displays the grid
							 row * size : (row + 1) * size] = channel_image
		scale = 1. / size
		plt.figure(figsize=(scale * display_grid.shape[1],
							scale * display_grid.shape[0]))
		plt.title(layer_name)
		plt.grid(False)
		plt.imshow(display_grid, aspect='auto', cmap='viridis')


def plot_learning_curves(history):
    plt.figure(figsize=(10,4), dpi=150)
    
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], 'bo--', marker='o', mfc='none')
    plt.plot(history.history['val_loss'], 'm-.',  linewidth=2)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.grid(linestyle='dotted')
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], 'bo--', marker='o', mfc='none')
    plt.plot(history.history['val_accuracy'], 'm-.',  linewidth=2)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.grid(linestyle='dotted')
        
    plt.tight_layout()
    



