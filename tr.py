import numpy
from keras import *
import numpy as np
#from PIL import array_to_image

#Part 1 - Build Model
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import LeakyReLU

#temp
labels = numpy.full((200,1),1)
label = numpy.full((200,1),2)
lab = numpy.concatenate((labels,label),a xis=0)



model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', activation='LeakyReLU()', input_shape=(128,128,3)))

model.add(Conv2D(64, (3, 3), activation='LeakyReLU()'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.35))

model.add(Conv2D(64, (3, 3), padding='same', activation='LeakyReLU()'))

model.add(Conv2D(64, (3, 3), activation='LeakyReLU()'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.35))

model.add(Conv2D(128, (3, 3), padding='same', activation='LeakyReLU()'))

model.add(Conv2D(128, (3, 3), activation='LeakyReLU()'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.35))

model.add(Conv2D(128, (3, 3), padding='same', activation='LeakyReLU()'))

model.add(Conv2D(128, (3, 3), activation='LeakyReLU()'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.35))

model.add(Flatten())

model.add(Dense(1024, activation='LeakyReLU()'))

model.add(Dropout(0.6))

model.add(Dense(39, activation='softmax'))


## Step 3 - Flattening
#model.add(Flatten())
#
## Step 4 - Full Connection
#model.add(Dense(units = 128, activation = 'LeakyReLU()'))
#model.add(Dense(units = 39, activation = 'softmax'))



# Compiling the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (128, 128),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('val',
                                            target_size = (128, 128),
                                            batch_size = 32,
                                            class_mode = 'categorical')


model.fit_generator(training_set,
                    steps_per_epoch = 40,
                    epochs = 80,
                    validation_data = test_set,
                    validation_steps = 500)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")



