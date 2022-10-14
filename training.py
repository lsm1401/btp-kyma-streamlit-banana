import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


#Load image for training

imgpath = '/Users/i551982/Desktop/Github/banana/data/'

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                 validation_split=0.3) #included in our dependencies
batch_size = 8

train_generator = train_datagen.flow_from_directory(imgpath,
                                                    target_size=(224,224),
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training',
                                                    shuffle=True)
val_generator = train_datagen.flow_from_directory(imgpath,
                                                  target_size=(224,224),
                                                  color_mode='rgb',
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  subset='validation',
                                                  shuffle=True)

#Load base model 
base_model=MobileNetV2(weights='imagenet', include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(512,activation='relu')(x) #dense layer 2
preds=Dense(3,activation='softmax')(x) #final layer with softmax activation
model = Model(inputs=base_model.input, outputs=preds)
model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
callbacks = [ModelCheckpoint('model_chkpt/weights.{epoch:02d}_{val_loss:.4f}_{val_accuracy:.4f}.h5')]

#Fit model 
model.fit_generator(generator=train_generator,
                    steps_per_epoch = train_generator.samples // batch_size,
                    validation_data=val_generator,
                    validation_steps = val_generator.samples // batch_size,
                    callbacks=callbacks,
                    epochs=20,
                    )

#Save model 
model.save("ripeness.h5")