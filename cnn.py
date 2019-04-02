from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# Initialize CNN
classifier = Sequential()

# 1st convolutional layer
classifier.add(Conv2D(128, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2nd convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flatten and form final classification layer 
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# parameters
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

# load training set
# data augmentation included: horizontal flipping
# shuffling included
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
training_set = train_datagen.flow_from_directory('G:\\Images\\mountains\\', target_size = (256, 256), batch_size = 32, shuffle = True, color_mode = "rgb", class_mode = 'binary')

# loading testing set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('G:\\Images\\mountainsTest\\', target_size = (256, 256), batch_size = 32, color_mode = "rgb", class_mode = 'binary')

# train CNN with training set
# hyperparameters set
# shuffles the batches
classifier.fit_generator(training_set, steps_per_epoch = 32, epochs = 3, validation_data = test_set, validation_steps = 2, shuffle = True)

# save model to disk
classifier.save("test.h5")