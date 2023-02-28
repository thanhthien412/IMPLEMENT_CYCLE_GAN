import os 
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from numpy import asarray
import tensorflow as tf
from numba import cuda
from model import define_discriminator, define_generator, define_composite_model, train
file=input()


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
device = cuda.get_current_device()
device.reset()
def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

def load_data(file,target_size=(128,128)):
	_,_,trainA,trainB=os.listdir(file)
	dataA,dataB=list(),list()
	trainA=os.path.join(file,trainA)
	trainB=os.path.join(file,trainB)
	for (imageA,imageB) in tqdm(zip(os.listdir(trainA),os.listdir(trainB))):
		imgA=load_img(trainA+'/'+imageA,target_size=target_size)
		imgB=load_img(trainB+'/'+imageB,target_size=target_size)
		imgA=img_to_array(imgA)
		imgB=img_to_array(imgB)
		dataA.append(imgA)
		dataB.append(imgB)
  
	return asarray(dataA),asarray(dataB)


dataA,dataB=load_data(file)
dataset=[dataA,dataB]
dataset=preprocess_data(dataset)

image_shape=(128,128,3)
d_model_A=define_discriminator(image_shape)
d_model_B=define_discriminator(image_shape)
g_model_AtoB=define_generator(image_shape,6)
g_model_BtoA=define_generator(image_shape,6)
c_model_AtoB=define_composite_model(g_model_AtoB,d_model_B,g_model_BtoA,image_shape)
c_model_BtoA=define_composite_model(g_model_BtoA,d_model_A,g_model_AtoB,image_shape)


train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=1,state=0)