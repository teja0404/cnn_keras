
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()


classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('keras training',
                                                 target_size = (64, 64),
                                                 batch_size = 3,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('keras test',
                                            target_size = (64, 64),
                                            batch_size = 3,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 2984,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 416)



training_set.class_indices
test_set.class_indices



import numpy as np
from keras.preprocessing import image
test_image=image.load_img("single_prediction\download (2).jpeg",target_size=(64,64))
test_image
numpy=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
training_set.class_indices
