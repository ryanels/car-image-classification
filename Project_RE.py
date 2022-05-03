#######################################
#Project file for Ryan Elson
#
########################################
import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization
import keras_tuner as kt
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import scipy.io as sio
import scipy.sparse.linalg as ll
import sklearn.preprocessing as skpp
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import csv
import time

np.random.seed(123)

#read in the data with x, y values for cropping, and class labels
temp = pd.read_csv("cardatasettrain.csv",dtype=str)
traindf = temp[["image", "Class", "Type", "Median MPG"]]
temp = pd.read_csv("cartest_withlabel.csv",dtype=str)
testdf = temp[["image", "Class", "Type", "Median MPG"]]

traindf['Class1'] = traindf["Class"].astype('int32')      #create new Class column as integers
temp1 = traindf.sort_values(['Class1']).groupby(["Class1"])["Class1"].size()   #get counts, grouping by class for histogram

#set which of the 3 classifications to use (Class, Type, Median MPG). Comment out 2. Use one class only.
ycol = 'Class'
#ycol = 'Type'
#ycol = 'Median MPG'

activation = tf.keras.activations.softmax
loss = tf.keras.losses.SparseCategoricalCrossentropy()
accuracy = tf.keras.metrics.sparse_categorical_accuracy
classlen = len(np.unique(traindf[ycol]))
classmode = 'sparse'

#early stopping; had set patience = 10, trying patience = 5
es= EarlyStopping(monitor ="val_loss", mode ="min", patience = 4, restore_best_weights = True)
numepochs = 50

############################################################
#  Create histograms of training data
############################################################

#histogram for body type of training data
plt.bar(x = temp1.index, height = temp1)
plt.title("Histogram (Make, Model, Year) - Training Data")
plt.ylabel("Count")
plt.xlabel("Class (Make, Model, Year)")
plt.xticks(visible = False)
plt.show()

#histogram for body type of training data
traindf['Type'].value_counts().plot(kind='bar')
plt.title("Histogram (Vehicle Type) - Training Data")
plt.ylabel("Count")
plt.xlabel("Class (Vehicle Type)")
plt.xticks(rotation = 45)
plt.show()

#histogram for Median MPG of training data
traindf['Median MPG'].value_counts().plot(kind='bar')
plt.title("Histogram (Median MPG) - Training Data")
plt.ylabel("Count")
plt.xlabel("Class (Above or Below Median MPG)")
plt.xticks(rotation = 0)
plt.show()

###############################################
# End of Histograms
###############################################
starttime = time.perf_counter()

with open('cardatasettrain.csv', 'r') as file:
    reader = csv.reader(file)
    cartrain = list(reader)        #cartrain data with x,y data and classes
with open('cardatasettest.csv', 'r') as file:
    reader = csv.reader(file)
    cartest = list(reader)         #cartest data with x, y data. No classes here
with open('cartest_withlabel.csv', 'r') as file:
    reader = csv.reader(file)
    cartest_labels = list(reader)    #cartest data with x,y data and classes
with open('classlabel.csv', 'r') as file:
    reader = csv.reader(file)
    classlabel = list(reader)   #all 196 different possible classes

train_dir = 'archive/cars_train_processed/'
test_dir = 'archive/cars_test_processed/'

#augment training data so it's rescaled, and randomly rotated, shifted, sheared, etc. Split of 0.25 for validation when used
train_datagen = ImageDataGenerator(rescale = 1./255.) #,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True) #, validation_split=.25)
test_datagen = ImageDataGenerator( rescale = 1.0/255.)

# Flow images in batches of 32 with our generators
#
# "flow from directory" expects all images of the same class to be in a subfolder with that class name. So a binary
# classification would have 'dog' and 'cat' folders under the 'train' folder, for example
#
# "flow from dataframe" allows us to read all files from a "train" folder for example, with a CSV that gives classes
# so images don't have to be in their own class folders
#
print('Train generator:')
train_generator = train_datagen.flow_from_dataframe(dataframe= traindf, directory= train_dir, x_col = 'image', y_col = ycol, shuffle= True, subset = "training", batch_size = 32, class_mode = classmode, color_mode='rgb', target_size = (224, 224))
# Flow validation images in batches of 20 using test_datagen generator

#set batch size to evenly divisible but validation is a percent of training data, not 8144. Tried 0.2 batch = 16, but not even, try 0.25 and batch = 4
#print('Validation generator:')
#validation_generator = train_datagen.flow_from_dataframe(dataframe= traindf, directory= train_dir, x_col = 'image', y_col = ycol, shuffle= True, subset= "validation", batch_size = 4, class_mode = classmode, color_mode='rgb', target_size = (224, 224))
print('Test generator:')
#set test generator batch size to 1 or something that the number of images divides evenly into
#for 8041 test images, options: 11, 731; 17, 473; 43, 187;
test_generator = test_datagen.flow_from_dataframe(dataframe = testdf, directory= test_dir, x_col = 'image', y_col = ycol, shuffle = False, batch_size = 17, class_mode = classmode, color_mode='rgb', target_size = (224, 224))

#Get step sizes
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
#STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

class_weights = class_weight.compute_class_weight(class_weight= 'balanced', classes = np.unique(traindf[ycol]), y=traindf[ycol])

####################################################
# Set up keras tuner
####################################################
def build_model(hp):
	# initialize the model along with the input shape and channel
	# dimension
	model = Sequential()
	inputShape = (224,224,3)
	chanDim = -1

# first CONV => RELU => POOL layer set
	model.add(Conv2D(64,
		#hp.Int("conv_1", min_value=32, max_value=96, step=32),
		(3, 3), padding="same", input_shape=inputShape))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))

# second CONV => RELU => POOL layer set
	model.add(Conv2D(#64,
		hp.Int("conv_2", min_value=64, max_value=128, step=32),
		(3, 3), padding="same"))
	model.add(Activation("relu"))
	model.add(BatchNormalization(axis=chanDim))
	model.add(MaxPooling2D(pool_size=(2, 2)))

# first (and only) set of FC => RELU layers
	model.add(Flatten())
	model.add(Dense(#64))
        hp.Int("dense_units", min_value=64,max_value=128, step=32)))
	model.add(Activation("relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	# softmax classifier
	model.add(Dense(classlen, activation = activation)) # sigmoid for binary (float 0.0 or 1.0), softmax for multi-class

# initialize the learning rate choices and optimizer
	lr = hp.Choice("learning_rate", values=[1e-3])
	opt = Adam(learning_rate=lr)
	# compile the model
	model.compile(optimizer=opt, loss= loss, metrics=["accuracy"])
    # return the model
	return model

# End of build_model function
#################################################

# set type of tuner (random, hyperband, or bayesian)
tuner = 'random'

# check if we will be using the hyperband tuner
if tuner == "hyperband":
    # instantiate the hyperband tuner object
    print("[INFO] instantiating a hyperband tuner object...")
    tuner = kt.Hyperband(
        build_model,
        objective="val_loss",
        max_epochs=numepochs,
        factor=3,
        seed=42,
        overwrite = True)
elif tuner == "random":
    # instantiate the random search tuner object
    print("[INFO] instantiating a random search tuner object...")
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=10,
        seed=42,
        overwrite = True)
# otherwise, we will be using the bayesian optimization tuner
else:
    # instantiate the bayesian optimization tuner object
    print("[INFO] instantiating a bayesian optimization tuner object...")
    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=10,
        seed=42,
        overwrite = True
    )

# perform the hyperparameter search
print("[INFO] performing hyperparameter search...")
tuner.search(
    x=train_generator, #y=trainY,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=test_generator,
    validation_steps = STEP_SIZE_TEST,
    #batch_size=32,
    callbacks=[es],
    epochs= numepochs
)

# grab the best hyperparameters
bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
#print("[INFO] optimal number of filters in conv_1 layer: {}".format(
#    bestHP.get("conv_1")))
print("[INFO] optimal number of filters in conv_2 layer: {}".format(
    bestHP.get("conv_2")))
print("[INFO] optimal number of units in dense layer: {}".format(
    bestHP.get("dense_units")))
print("[INFO] optimal learning rate: {:.4f}".format(
    bestHP.get("learning_rate")))

# build the best model and train it
print("[INFO] training the best model...")
model = tuner.hypermodel.build(bestHP)

H = model.fit(x=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data = test_generator,
                    validation_steps = STEP_SIZE_TEST,
                    callbacks = [es],
                    epochs= numepochs,
                    class_weight = class_weights,
                    verbose = 1)
print('History (H):', H.history)
model.save("saved_model_tuned")

# evaluate the network
print("[INFO] evaluating network...")
test_generator.reset()
pred = model.predict(x= test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames

results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

results.to_csv("results_tuned.csv",index=False)

#################################################
# CNN model, working but parameters not tuned
#################################################
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(224,224,3)))  #shape is image size (224x224) and colors (1 if grayscale, 3 if rgb)
model.add(Activation('relu'))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classlen, activation= activation))  #sigmoid for binary (float 0.0 or 1.0), softmax for multi-class
model.compile(optimizer='rmsprop',loss= loss ,metrics=[accuracy])
#loss is tf.keras.losses.BinaryCrossentropy(from_logits = True) for binary
#loss is tf.keras.losses.SparseCategoricalCrossentropy for multiclass with integer labels
#accuracy is tf.keras.metrics.BinaryAccuracy for binary
#accuracy is tf.keras.metrics.SparseCategoricalAccuracy for multiclass with integer labels

history = model.fit(x=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data = test_generator,
                    validation_steps = STEP_SIZE_TEST,
                    callbacks = [es],
                    epochs= numepochs,
                    class_weight = class_weights)

print('History:', history.history)
model.save("saved_model")
# It can be used to reconstruct the model identically.
#reconstructed_model = keras.models.load_model("my_model")

#print('Model Evaluate results: for Validation Generator')
#model.evaluate(x=validation_generator, steps = STEP_SIZE_VALID) #, steps=STEP_SIZE_TEST)
print('Model Evaluate results: for Test Generator')
model.evaluate(x=test_generator, steps = STEP_SIZE_TEST)

test_generator.reset()
pred=model.predict(x= test_generator,
    steps=STEP_SIZE_TEST,
    verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

#getting stuck here. class_indices is not valid? But then ran fine the next time...

#labels = (train_generator.class_indices)
labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames

results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

results.to_csv("results.csv",index=False)
'''

stoptime = time.perf_counter()
runtime = stoptime-starttime
print(f"Runtime for {numepochs} epochs for {ycol} classification: {runtime:.2f} seconds = {runtime/60:.2f} minutes")

#imagename = cartest_labels[1][6]
#print(os.getcwd())

#image = Image.open(test_dir + imagename)
#plt.imshow(image)
#plt.show()