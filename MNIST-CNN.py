#### Initialization ####
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras_sequential_ascii import keras2ascii
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
import sklearn.metrics as skm
np.random.seed(25)

#### Loading the MNIST Dataset ####
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)

#### Show an example of the dataset ####
plt.imshow(X_train[0], cmap='gray')
plt.title('Class '+ str(y_train[0]))
plt.show()

#### Pre-processing of the dataset ####
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train/=255
X_test/=255
number_of_classes = 10
Y_train = np_utils.to_categorical(y_train, number_of_classes)
Y_test = np_utils.to_categorical(y_test, number_of_classes)

#### Design the Neural Network ####
# Each Convolutional layer consists of 3 step
# 1. Convolution
# 2. Activation
# 3. Polling
#After that make a fully connected network This fully connected network gives ability to the CNN to classify the samples
model = Sequential()
#Convolutional layers
model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)
model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
# Fully connected layer
BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
BatchNormalization()
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))
#### network architecture#######
keras2ascii(model)
model.summary() 

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                         height_shift_range=0.08, zoom_range=0.08)
test_gen = ImageDataGenerator()
batch_size=1024
train_generator = gen.flow(X_train, Y_train, batch_size=batch_size)
test_generator = test_gen.flow(X_test, Y_test, batch_size=batch_size)
#### you can switch between the normal data and augmented data ####
#history=model.fit_generator(train_generator, steps_per_epoch=y_train.size//batch_size, epochs=20, 
             #       validation_data=test_generator, validation_steps=y_test.size//batch_size)
history=model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=20, validation_data=(X_test, Y_test))
### Accuracy####
score = model.evaluate(X_test, Y_test)
print()
print('Test accuracy: ', score[1])
#### Predication Result #####
Y_Predict=model.predict(X_test)

### plot the performance curves ###

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

### Generate the one_hot prediction vector ####
Y_Predict_one_hot=np.zeros(np.shape(Y_Predict))
indx=0
for i in (Y_Predict):
    Y_Predict_one_hot[indx,np.argmax(i)]=1
    indx=indx+1

### Generate the Classification accuracy and report####
print(skm.classification_report(Y_test,Y_Predict_one_hot))
print(skm.accuracy_score(Y_test,Y_Predict_one_hot))

### Generate the Confusion matrix and Accuracy for each Class ###
Confusion_matrix=np.zeros([10,10])
indx=0
for i in (Y_Predict_one_hot):
    j=Y_test[indx,:]
    Confusion_matrix[np.argmax(j),np.argmax(i)]+=1
    indx=indx+1
print(Confusion_matrix.astype(int))
indx=0
for i in Confusion_matrix.astype(int):
    print("The Accuracy for class",int(indx),"is:",i[indx]/sum(i)*100)
    indx=indx+1
#### 