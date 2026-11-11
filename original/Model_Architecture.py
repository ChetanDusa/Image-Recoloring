import cv2 as cv
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate,UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError
#from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab
path = r"C:\Users\Kondu\OneDrive\Desktop\dataset(color images)"
train_datagen = ImageDataGenerator(rescale=1./255)
train = train_datagen.flow_from_directory(path, target_size=(256,256), class_mode=None, batch_size=32)

x = []
y = []


for i in range(len(train)):
    batch = train[i]
    for img in batch:
        try:
            lab = rgb2lab(img)
            x.append(lab[:,:,0])
            y.append(lab[:,:,1:] / 128)
        except Exception as e:
            print(f'Error: {e}')

x = np.array(x)
y = np.array(y)
x = x.reshape(x.shape + (1,))

print(x.shape)
print(y.shape)


model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(512, (3,3), activation='relu', padding='same',strides=2))
# model.add(BatchNormalization())
# model.add(Conv2D(1024, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())

model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='adam', loss='mse' , metrics=['accuracy'])
model.summary()
# model.fit(x,y,validation_split=0.1,batch_size=25,epochs=300)
# model.save('colorize_autoencoder(unwanted).h5')
checkpoint_path = r"C:\Users\Kondu\project(iitSoC)\models\checkpoint(first-100).keras"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    verbose=1
)

model.fit(x,y,validation_split=0.07,batch_size=25,epochs=400,callbacks=[cp_callback])
