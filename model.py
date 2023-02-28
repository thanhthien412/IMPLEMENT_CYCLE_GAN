from tabnanny import verbose
import numpy as np
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from sklearn.utils import shuffle
from tqdm import tqdm
from instancenormalization import InstanceNormalization
from matplotlib import pyplot





def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64: 4x4 kernel Stride 2x2
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128: 4x4 kernel Stride 2x2
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256: 4x4 kernel Stride 2x2
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512: 4x4 kernel Stride 2x2 
    # Not in the original paper. Comment this block if you want.
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer : 4x4 kernel but Stride 1x1
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# define model
	model = Model(in_image, patch_out)
	# compile model
	model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model


def resnet_block(n_filters, input_layer):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# first convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# second convolutional layer
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	# concatenate merge channel-wise with input layer
	g = Concatenate()([g, input_layer])
	return g

def define_generator(image_shape, n_resnet=9):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# c7s1-64
	g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d128
	g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# d256
	g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# R256
	for _ in range(n_resnet):
		g = resnet_block(256, g)
	# u128
	g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# u64
	g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	# c7s1-3
	g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	out_image = Activation('tanh')(g)
	# define model
	model = Model(in_image, out_image)
	return model
 
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	g_model_1.trainable = True
	# mark discriminator and second generator as non-trainable
	d_model.trainable = False
	g_model_2.trainable = False
    
	# adversarial loss
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity loss
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# cycle loss - forward
	output_f = g_model_2(gen1_out)
	# cycle loss - backward
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
    
	# define model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	
    # define the optimizer
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], 
               loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model

# load and prepare training images
def load_real_samples(filename):
	# load the dataset
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

def generate_real_samples(dataset, n_samples, patch_shape):
	ix = randint(0, dataset.shape[0], n_samples)
	X = dataset[ix]
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return X, y
def generate_fake_samples(g_model, dataset, patch_shape):
	X = g_model.predict(dataset,verbose=0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def save_models(step, g_model_AtoB, g_model_BtoA):
	# save the first generator model
	filename1 = 'weight_AtoB/g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	# save the second generator model
	filename2 = 'weight_BtoA/g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

# periodically generate images using the save model and plot input and output images
def summarize_performance(step, g_model, trainX, name, n_samples=5):
	# select a sample of input images
	X_in, _ = generate_real_samples(trainX, n_samples, 0)
	# generate translated images
	X_out, _ = generate_fake_samples(g_model, X_in, 0)
	# scale all pixels from [-1,1] to [0,1]
	X_in = (X_in + 1) / 2.0
	X_out = (X_out + 1) / 2.0
	# plot real images
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_in[i])
	# plot translated image
	for i in range(n_samples):
		pyplot.subplot(2, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_out[i])
	# save plot to file
	filename1 = 'train_%s/generated_plot_%06d.png' % (name,(step+1))
	pyplot.savefig(filename1)
	pyplot.close()


def update_image_pool(pool,image,max_size=50):
    selected=list()
    if len(pool)==0:
        pool.append(image)
        selected.append(image)
    elif len(pool)==max_size:
        ix=randint(image)
        selected.append(pool[ix])
        pool[ix]=image
    else:
        ix=randint(0,len(pool))
        selected.append(pool[ix])
        pool.append(image)
    
    return asarray(selected)
        
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, epochs=1,state=0):
    n_batch=1
    
    n_patch=d_model_A.output_shape[1]
    
    trainA,trainB=dataset
    
    poolA, poolB=list(),list()
    
    batch_per_epo=int(len(trainA)/n_batch)
    
    num=len(trainA)
    
    if(state!=0):
        g_model_AtoB.load_weights('weight_AtoB/g_model_AtoB_%06d.h5'%(state))
        g_model_BtoA.load_weights('weight_BtoA/g_model_BtoA_%06d.h5'%(state))
        print('SUCCESSFULLY UPDATE')
        print('='*30)
        
    for epoch in range(state,epochs):
        trainA,trainB=shuffle(trainA,trainB,random_state=epochs)
        totall_loss_dA1=0.0
        totall_loss_dA2=0.0
        totall_loss_dB1=0.0
        totall_loss_dB2=0.0
        totall_loss_g1=0.0
        totall_loss_g2=0.0
        
        for batch in tqdm(range(batch_per_epo)):
            y_realA=ones((1,n_patch,n_patch,1))
            y_realB=y_realA.copy()
            # generate a batch of fake samples using both B to A and A to B generators
            X_realA, X_realB=np.expand_dims(trainA[batch],0),np.expand_dims(trainB[batch],0)
            X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB , n_patch)
            X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA , n_patch)
            # update fake images in the pool. Remember that the paper suggstes a buffer of 50 images
            X_fakeA = update_image_pool(poolA, X_fakeA)
            X_fakeB = update_image_pool(poolB, X_fakeB)
            X_fakeA=X_fakeA[0]
            X_fakeB=X_fakeB[0]
            # update generator B->A via the composite model
            totall_loss_g2  += c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])[0]
            # update discriminator for A -> [real/fake]
            totall_loss_dA1 += d_model_A.train_on_batch(X_realA, y_realA)
            totall_loss_dA2 += d_model_A.train_on_batch(X_fakeA, y_fakeA)
            # update generator A->B via the composite model
            totall_loss_g1  += c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])[0]
            # update discriminator for B -> [real/fake]
            totall_loss_dB1 += d_model_B.train_on_batch(X_realB, y_realB)
            totall_loss_dB2 += d_model_B.train_on_batch(X_fakeB, y_fakeB)
            
        print('Epoch: %d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (epoch+1, totall_loss_dA1/num,totall_loss_dA2/num, totall_loss_dB1/num,totall_loss_dB2/num,totall_loss_g1/num,totall_loss_g2/num))
        
        if(epoch+1%10==0):
            summarize_performance(epoch,g_model_AtoB,trainA,'AtoB')
            
            summarize_performance(epoch,g_model_BtoA,trainB,'BtoA')
            
            save_models(epoch,g_model_AtoB,g_model_BtoA)
            
            