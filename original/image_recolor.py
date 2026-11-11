import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img
from skimage.transform import resize
from skimage.io import imsave, imshow
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))
#Replace your path to model insted of 'C:\Users\Kondu\project(iitSoC)\models\Image colorizer.keras'
model = tf.keras.models.load_model(r'C:\Users\Kondu\project(iitSoC)\models\Image colorizer.keras',
                                   custom_objects={'mse': mse},
                                   compile=True)

img1_color=[]
image_path=r"C:\Users\Kondu\OneDrive\Desktop\dataset(color images-2)\temp\prabhas-3.jpg"#Path to image
ori=img_to_array(load_img(image_path))
ori_cv=cv.imread(image_path)

img1 = resize(ori ,(256,256))
img1_color.append(img1)

img1_color = np.array(img1_color, dtype=float)
img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
img1_color = img1_color.reshape(img1_color.shape+(1,))

output1 = model.predict(img1_color)
output1 = output1*128

result = np.zeros((256, 256, 3))
result[:,:,0] = img1_color[0][:,:,0]
result[:,:,1:] = output1[0]

result = np.zeros((256, 256, 3))
result[:,:,0] = img1_color[0][:,:,0]
result[:,:,1:] = output1[0]
result=lab2rgb(result)
result=resize(result ,ori_cv.shape)
imshow(result)
plt.title('Image from skimage')
plt.axis('off')
plt.show()
