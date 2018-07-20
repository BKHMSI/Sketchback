
from __future__ import print_function
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.misc import imresize
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Activation, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, Nadam
from keras.utils import np_utils, plot_model
from keras import objectives, layers
from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input
from keras import backend as K

np.random.seed(1337)  # for reproducibility


# CelebA Faces: 72x88 200K Images
# ZuBuD Buildings: 120x160 3K Images
# CUHK Faces: 80x112 88 Images

m = 205
n = 282
sketch_dim = (m,n)
img_dim = (m,n,3)
num_images = 3000
num_epochs = 20
batch_size = 5
file_names = []

CelebA_SKETCH_PATH = '~/Project/CelebA_Sketch'
CelebA_IMAGE_PATH = '~/Project/img_align_celeba'

BUILDING_SKETCH_PATH = '~/Project/ZuBuD_Sketch_Aug'
BUILDING_IMAGE_PATH = '~/Project/ZuBuD_Aug'

CUHK_SKETCH_PATH = '~/Project/CUHK_Sketch'
CUHK_IMAGE_PATH = '~/Project/CUHK'


base_model = vgg16.VGG16(weights='imagenet', include_top=False)
vgg = Model(input=base_model.input, output=base_model.get_layer('block2_conv2').output)


def load_file_names(path):
    return os.listdir(path)

def sub_plot(x,y,z):
    fig = plt.figure()
    a = fig.add_subplot(1,3,1)
    imgplot = plt.imshow(x, cmap='gray')
    a.set_title('Sketch')
    plt.axis("off")
    a = fig.add_subplot(1,3,2)
    imgplot = plt.imshow(z)
    a.set_title('Prediction')
    plt.axis("off")
    a = fig.add_subplot(1,3,3)
    imgplot = plt.imshow(y)
    a.set_title('Ground Truth')
    plt.axis("off")
    plt.show()

def imshow(x, gray=False):
    plt.imshow(x, cmap='gray' if gray else None)
    plt.show()


def get_batch(idx, X = True, Y = True, W = True, dataset='zubud'):
    
    global file_names

    X_train = np.zeros((batch_size, m, n), dtype='float32')
    Y_train = np.zeros((batch_size, m, n, 3), dtype='float32')
    F_train = None
    
    if dataset == 'zubud':
        x_path = BUILDING_SKETCH_PATH
        y_path = BUILDING_IMAGE_PATH
    elif dataset == 'cuhk':
        x_path = CUHK_SKETCH_PATH
        y_path = CUHK_IMAGE_PATH
    else:
        x_path = CelebA_SKETCH_PATH
        y_path = CelebA_IMAGE_PATH
    
    if len(file_names) == 0:
        file_names = load_file_names(x_path)
        
    if X:
        # Load Sketches
        for i in range(batch_size):
            file = os.path.join(x_path, file_names[i+batch_size*idx])
            img = cv.imread(file,0)
            img = imresize(img, sketch_dim)
            img = img.astype('float32')
            X_train[i] = img / 255.
            
    if Y:
        # Load Ground-truth Images
        for i in range(batch_size):
            file = os.path.join(y_path, file_names[i+batch_size*idx])
            img = cv.imread(file)
            img = imresize(img, img_dim)
            if dataset != 'zubud':
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = img.astype('float32')
            Y_train[i] = img / 255.
    
    if W:
        F_train = get_features(Y_train)
    
    X_train = np.reshape(X_train, (batch_size, m, n, 1))
    return X_train, Y_train, F_train


def get_features(Y):
    Z = deepcopy(Y)
    Z = preprocess_vgg(Z)
    features = vgg.predict(Z, batch_size = 5, verbose = 0)
    return features


def preprocess_vgg(x, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}
    x = 255. * x
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
        # Zero-center by mean pixel
        x[:, 0, :, :] = x[:, 0, :, :] - 103.939
        x[:, 1, :, :] = x[:, 1, :, :] - 116.779
        x[:, 2, :, :] = x[:, 2, :, :] - 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, :, 0] = x[:, :, :, 0] - 103.939
        x[:, :, :, 1] = x[:, :, :, 1] - 116.779
        x[:, :, :, 2] = x[:, :, :, 2] - 123.68
    return x

def preprocess_VGG(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}
    # x has pixels intensities between 0 and 1
    x = 255. * x
    norm_vec = K.variable([103.939, 116.779, 123.68])
    if dim_ordering == 'th':
        norm_vec = K.reshape(norm_vec, (1,3,1,1))
        x = x - norm_vec
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        norm_vec = K.reshape(norm_vec, (1,1,1,3))
        x = x - norm_vec
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def feature_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

def pixel_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred))) + 0.00001*total_variation_loss(y_pred)

def total_variation_loss(y_pred):
    if K.image_data_format() == 'channels_first':
        a = K.square(y_pred[:, :, :m - 1, :n - 1] - y_pred[:, :, 1:, :n - 1])
        b = K.square(y_pred[:, :, :m - 1, :n - 1] - y_pred[:, :, :m - 1, 1:])
    else:
        a = K.square(y_pred[:, :m - 1, :n - 1, :] - y_pred[:, 1:, :n - 1, :])
        b = K.square(y_pred[:, :m - 1, :n - 1, :] - y_pred[:, :m - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def generator_model(input_img):

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(64, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    res = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])
    res = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.add([x, res])

    # Decoder
    res = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv1')(encoded)
    x = layers.add([encoded, res])
    res = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])
    res = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])

    x = Conv2D(128, (2, 2), activation='relu', padding='same', name='block6_conv1')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block7_conv1')(x)
    res = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])
    res = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])

    x = Conv2D(64, (2, 2), activation='relu', padding='same', name='block8_conv1')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block9_conv1')(x)
    res = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])
    res = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])

    x = Conv2D(32, (2, 2), activation='relu', padding='same', name='block10_conv1')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block11_conv1')(x)
    res = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.add([x, res])
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    return decoded

def feat_model(img_input):
    # extract vgg feature
    vgg_16 = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=None)
    # freeze VGG_16 when training
    for layer in vgg_16.layers:
        layer.trainable = False
    
    vgg_first2 = Model(input=vgg_16.input, output=vgg_16.get_layer('block2_conv2').output)
    Norm_layer = Lambda(preprocess_VGG)
    x_VGG = Norm_layer(img_input)
    feat = vgg_first2(x_VGG)
    return feat

def full_model(summary = True):
    input_img = Input(shape=(m, n, 1))
    generator = generator_model(input_img)
    feat = feat_model(generator)
    model = Model(input=input_img, output=[generator, feat], name='architect')
    model.summary()
    return model

def train_faces(weights=None):

    model = full_model()
    optim = Adam(lr=1e-4,beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(loss=[pixel_loss, feature_loss], loss_weights=[1, 0.01], optimizer=optim)

    if weights is not None:
        model.load_weights(weights)

    for epoch in range(num_epochs):
        num_batches = num_images // batch_size

        for batch in range(num_batches):
            X,Y,W = get_batch(batch, dataset='zubud')
            loss = model.train_on_batch(X, [Y, W])
            print("Loss in Epoch # ",epoch,"| Batch #", batch, ":", loss)

        model.save_weights("weights_%d" % epoch)


def predict(batch, i, weights):
    model = full_model()
    model.load_weights(weights)
    X, T, _ = get_batch(batch, Y = True, W = False, dataset='zubud')
    Y, W = model.predict(X[:i])
    x = X[i].reshape(m,n)
    y = Y[i]
    sub_plot(x, T[i], y)

def sketchback(image, weights):
    model = full_model()
    model.load_weights(weights)
    sketch = cv.imread(image, 0)
    sketch = imresize(sketch, sketch_dim)
    sketch = sketch / 255.
    sketch = sketch.reshape(1,m,n,1)
    result, _ = model.predict(sketch)
    imshow(result[0])
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    imgplot = plt.imshow(sketch[0].reshape(m,n), cmap='gray')
    a.set_title('Sketch')
    plt.axis("off")
    a = fig.add_subplot(1,2,2)
    imgplot = plt.imshow(result[0])
    a.set_title('Prediction')
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    sketchback("face7.jpg", "weights_faces")