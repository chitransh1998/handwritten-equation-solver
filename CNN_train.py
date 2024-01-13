#Code to build the model and train it

from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img
from numpy import array
from keras import regularizers
import cv2


#initializing the model
model= Sequential()

#adding convolutional layers and pooling layers 
model.add(Convolution2D(32,3,3, input_shape=(45,45,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3, input_shape=(45,45,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#Dropout layers to reduce over-fitting
model.add(Dropout(0.3)) 
model.add(Flatten())

#Now two hidden(dense) or fully-connected layers are added:
model.add(Dense(output_dim = 300, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.4))#again for regularization
model.add(Dense(output_dim = 300, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))


model.add(Dropout(0.3))#last one lol

model.add(Dense(output_dim = 150, activation = 'relu',
                kernel_regularizer=regularizers.l2(0.01)))

#output layer
model.add(Dense(output_dim = 24, activation = 'sigmoid'))


#Now compile it
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#Now generate training and test sets from folders

train_datagen=ImageDataGenerator(
                                   rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.,
                                   horizontal_flip = False,
                                   validation_split=0.2 
                                 )


training_set=train_datagen.flow_from_directory("dataset",
                                               target_size = (45,45),
                                               color_mode='grayscale',
                                               batch_size=10,
                                               class_mode='categorical',subset='training')

test_set=train_datagen.flow_from_directory("dataset", 
                                               target_size = (45,45),
                                               color_mode='grayscale',
                                               batch_size=10,
                                               class_mode='categorical',subset='validation')






#finally, start training
model.fit_generator(training_set,
                         samples_per_epoch = 1956,
                         nb_epoch = 80,
                         validation_data = test_set,
                         nb_val_samples = 320)


#saving the weights
model.save_weights("weights.hdf5",overwrite=True)

#saving the model itself in json format:
model_json = model.to_json()
with open("model.json", "w") as model_file:
    model_file.write(model_json)
print("Model has been saved.")









