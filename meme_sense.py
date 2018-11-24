"""
Finetuning MovileNetV2 to distiguish between an religious image and not an religious image.
"""

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.applications.mobilenetv2 import MobileNetV2
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary', shuffle=True)

validation_generator = test_datagen.flow_from_directory(
        'data/val',
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary', shuffle=True)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model

top_model = Sequential()
top_model.add(base_model)
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(2048, activation='relu'))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dense(1, activation='sigmoid'))



top_model.compile(loss='binary_crossentropy', optimizer=SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
top_model.summary()


checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)

top_model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=25,
        validation_data=validation_generator,
        validation_steps=80, callbacks=[checkpointer])

