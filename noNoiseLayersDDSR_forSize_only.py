########################################################################################################################
########################################################################################################################
# Authors: David Schwartz
#
# deate: 1/27/2022
# This implements the data dependent stochastic resonance network
# ideally this will be more robust against adversarial and gaussian noise, but we'll have to do science to find out.
# For now we'll use a vgg like CNN with fewer parameters than the full vgg.
########################################################################################################################
import gc
import sys
from sys import platform
from venv import create
if platform == "linux" or platform == "linux2":
    tmpPathPrefix = '/tmp/'
elif platform == "win32":
    tmpPathPrefix = 'A:/tmp/'

from unittest.mock import call
# if ('linux' not in sys.platform):
#     import pyBILT_common as pyBILT
# from numba import cuda 
# device = cuda.get_current_device()

# import infotheory
# import lempel_ziv_complexity
import scipy
import pickle
from itertools import chain
# import tfMI
# from sklearn.metrics.cluster import adjusted_mutual_info_score, mutual_info_score
# from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, trange
# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras import Model
# from tensorflow.keras import regularizers
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.utils import safe_indexing, indexable

from sklearn.utils import _safe_indexing as safe_indexing, indexable
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, TimeDistributed, Conv1D, BatchNormalization, Concatenate
from tensorflow.random import Generator
# tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import tensorflow as tf

# tf.keras.utils.set_random_seed(1)
# tf.config.experimental.enable_op_determinism()
# from keras.callbacks import ModelCheckpoint#, TensorBoard
# from keras.callbacks import Callback
import os
import cv2
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras import regularizers, losses, utils
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from tqdm import tqdm, trange
import numpy as np
import time
import re, random, collections
from collections import defaultdict
from numpy import linalg as LA

import itertools

########################################################################################################################


#define RNG layer
@tf.keras.utils.register_keras_serializable()
class rngSource(tf.keras.layers.Layer):
    def __init__(self, std_dev=0.5, **kwargs):
        self.std_dev = std_dev
        super(rngSource, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        config["std_dev"] = self.std_dev
        return config
    def call(self, inputs):
        return tf.cast(K.random_normal(shape=tf.shape(inputs), mean=0, stddev=self.std_dev, seed=42),dtype=tf.float32)


defaultLossThreshold = 0.00001
defaultDropoutRate = 0

class DDSR(object):
    def __init__(self, input_dimension, output_dimension, number_of_classes=2, optimizer=None, dual_outputs=False,
        loss_threshold=defaultLossThreshold, patience=10, dropout_rate=defaultDropoutRate, reg=None, stochasticResonatorOn=False,
        max_relu_bound=None, adv_penalty=0.01, adv_penalty_l1=0.01, explicitDataDependence=False, usingAdvReg=False, freezeWeights=False, verbose=False):

        self.buildModel(input_dimension, output_dimension, number_of_classes=number_of_classes, dual_outputs=dual_outputs,
                        loss_threshold=loss_threshold, patience=patience, dropout_rate=dropout_rate, stochasticResonatorOn=stochasticResonatorOn,
                        max_relu_bound=max_relu_bound, adv_penalty=adv_penalty, adv_penalty_l1=adv_penalty_l1, explicitDataDependence=explicitDataDependence, usingAdvReg=usingAdvReg,
                        optimizer=optimizer, reg=reg, freezeWeights=freezeWeights, verbose=verbose)


    def buildModel(self, input_dimension, output_dimension, number_of_classes=2, optimizer=None, dual_outputs=False,
        loss_threshold=defaultLossThreshold, patience=10, dropout_rate=defaultDropoutRate, 
        max_relu_bound=None, adv_penalty=0.01, adv_penalty_l1=0.01, explicitDataDependence=False, usingAdvReg=False, reg=None, 
        stochasticResonatorOn=False, freezeWeights=False, verbose=False):
        self.input_dimension, self.output_dimension = input_dimension, np.copy(output_dimension)
        self.advPenalty = np.copy(adv_penalty)

        self.loss_threshold, self.number_of_classes = np.copy(loss_threshold), np.copy(number_of_classes)
        self.dropoutRate, self.max_relu_bound = np.copy(dropout_rate), np.copy(max_relu_bound)
        self.patience = np.copy(patience)
        self.explicitDataDependence = explicitDataDependence
        self.dualOutputs = dual_outputs
        self.stochasticResonatorOn = stochasticResonatorOn
        standardDev = 0.5 if (self.stochasticResonatorOn) else 0

        ##self.learning_rate, self.learning_rate_drop = np.copy(learning_rate), np.copy(learning_rate_drop)
        self.image_size = 32
        self.num_channels = 3
        self.num_labels = np.copy(number_of_classes)
        self.penaltyCoeff = np.copy(adv_penalty)
        self.penaltyCoeffL1 = np.copy(adv_penalty_l1)

        #set up the RNG
        # self.globalRNGs = tf.random.Generator.from_seed(42).split(5)

        #decide an activation function
        self.chosenActivation = "relu" if max_relu_bound is not None else "tanh"

        print(self.input_dimension)
        print(input_dimension)
        # define input layer
        self.inputLayer = layers.Input(shape=self.input_dimension)
        previousLayer = self.inputLayer
        
        #define the adversarial main input layer (only used in training)
        self.advInputLayer = layers.Input(shape=self.input_dimension)
        previousAdvLayer = self.advInputLayer

        #define the hidden layers
        self.hiddenLayers = dict()
        self.hiddenAdvLayers = dict()
        self.poolingLayers = dict()
        self.resonators = dict()
        self.couplingLayers = dict()
        
        #define hidden layer outputs
        self.hiddenModelOutputs, self.hiddenAdvModelOutputs = dict(), dict()

        #layer 0    
        self.hiddenLayers[0] = layers.Conv2D(32, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[0] = self.hiddenLayers[0]
        previousLayerInput = previousLayer
        previousLayer = self.hiddenLayers[0](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[0](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)

        #layer zero's stochastic resonator
        #instantiate a new rng
        
        # #if we are not explicitly forcing data dependence with the RNG processing
        # if not self.explicitDataDependence:
        #     self.resonators[0] = rngSource(standardDev)(previousLayerInput)    
        #     #for now, the coupling weights will be constant (and learned not as a function of the particular input, but of the statistics of the input data and the neural network as it learns)        
        #     self.couplingLayers[0] = layers.Conv2D(32, (3,3), activation='linear', padding='same')(self.resonators[0])
        #     #compute output of the combination of the weighted RNG samples with the previous layer's output
        #     previousLayer = previousLayer + self.couplingLayers[0]
        #     previousAdvLayer = previousAdvLayer + self.couplingLayers[0]

        # else:
        #     self.resonators[0] = rngSource(standardDev)(previousLayer)
        #     #for now, the coupling weights will be constant (and learned not as a function of the particular input, but of the statistics of the input data and the neural network as it learns)        
        #     self.couplingLayers[0]  = layers.Conv2D(32, (3,3), activation='linear', padding='same')(previousLayerInput)
        #     #compute output of the combination of the weighted RNG samples with the previous layer's output
        #     previousLayer = previousLayer + tf.keras.layers.Multiply()([self.couplingLayers[0], self.resonators[0]])
        #     previousAdvLayer = previousAdvLayer + tf.keras.layers.Multiply()([self.couplingLayers[0], self.resonators[0]])
        

        



        # #dropout after noise
        # if (self.dropoutRate > 0):
        #     previousLayer = Dropout(dropout_rate)(previousLayer)
        #     previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)
        
        #organize intermediate outputs
        self.hiddenModelOutputs[0] = previousLayer
        self.hiddenAdvModelOutputs[0] = previousAdvLayer

        #layer 1
        self.hiddenLayers[1] = layers.Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[1] = self.hiddenLayers[1]
        previousLayerInput = previousLayer
        previousLayer = self.hiddenLayers[1](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[1](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        
        # #layer one's stochastic resonator
        # #instantiate a new rng
        # #if we are explicitly forcing data dependence with the RNG processing
        # if not self.explicitDataDependence:
        #     self.resonators[1] = rngSource(standardDev)(previousLayerInput)
        #     #for now, the coupling weights will be constant (and learned not as a function of the particular input, but of the statistics of the input data and the neural network as it learns)        
        #     self.couplingLayers[1] = layers.Conv2D(64, (3,3), activation='linear', padding='same')(self.resonators[1])
        #     #compute output of the combination of the weighted RNG samples with the previous layer's output
        #     previousLayer = previousLayer + self.couplingLayers[1]
        #     previousAdvLayer = previousAdvLayer + self.couplingLayers[1]

        # else:
        #     self.resonators[1] = rngSource(standardDev)(previousLayer)
        #     #for now, the coupling weights will be constant (and learned not as a function of the particular input, but of the statistics of the input data and the neural network as it learns)        
        #     self.couplingLayers[1] = layers.Conv2D(64, (3,3), activation='linear', padding='same')(previousLayerInput)
        #     #compute output of the combination of the weighted RNG samples with the previous layer's output
        #     previousLayer = previousLayer + tf.keras.layers.Multiply()([self.couplingLayers[1], self.resonators[1]])
        #     previousAdvLayer = previousAdvLayer + tf.keras.layers.Multiply()([self.couplingLayers[1], self.resonators[1]])
    
        # #droput after nois]e
        # if (self.dropoutRate > 0):
        #     previousLayer = Dropout(dropout_rate)(previousLayer)
        #     previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #pooling
        self.poolingLayers[0] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[0](previousLayer)
        previousAdvLayer = self.poolingLayers[0](previousAdvLayer)
        previousLayerInput = previousLayer

        #organize intermediate outputs
        self.hiddenModelOutputs[1] = previousLayer
        self.hiddenAdvModelOutputs[1] = previousAdvLayer

        #layer 2
        self.hiddenLayers[2] = layers.Conv2D(128, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[2] = self.hiddenLayers[2]
        previousLayer = self.hiddenLayers[2](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[2](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[2] = previousLayer
        self.hiddenAdvModelOutputs[2] = previousAdvLayer
        
        # #layer two's stochastic resonator
        # #instantiate a new rng
        # #if we are explicitly forcing data dependence with the RNG processing
        # if not self.explicitDataDependence:
        #     self.resonators[2] = rngSource(standardDev)(previousLayerInput)
        #     #for now, the coupling weights will be constant (and learned not as a function of the particular input, but of the statistics of the input data and the neural network as it learns)        
        #     self.couplingLayers[2] = layers.Conv2D(128, (3,3), activation='linear', padding='same')(self.resonators[2])
        #     #compute output of the combination of the weighted RNG samples with the previous layer's output
        #     previousLayer = previousLayer + self.couplingLayers[2]
        #     previousAdvLayer = previousAdvLayer + self.couplingLayers[2]

        # else:
        #     self.resonators[2] = rngSource(standardDev)(previousLayer)
        #     #for now, the coupling weights will be constant (and learned not as a function of the particular input, but of the statistics of the input data and the neural network as it learns)        
        #     self.couplingLayers[2] = layers.Conv2D(128, (3,3), activation='linear', padding='same')(previousLayerInput)
        #     #compute output of the combination of the weighted RNG samples with the previous layer's output
        #     previousLayer = previousLayer + tf.keras.layers.Multiply()([self.couplingLayers[2], self.resonators[2]])
        #     previousAdvLayer = previousAdvLayer + tf.keras.layers.Multiply()([self.couplingLayers[2], self.resonators[2]])

        #organize intermediate outputs
        self.hiddenModelOutputs[2] = previousLayer
        self.hiddenAdvModelOutputs[2] = previousAdvLayer
        
        #droput after noise
        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 3
        self.hiddenLayers[3] = layers.Conv2D(256, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[3] = self.hiddenLayers[3]
        previousLayerInput = previousLayer
        previousLayer = self.hiddenLayers[3](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[3](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[3] = previousLayer
        self.hiddenAdvModelOutputs[3] = previousAdvLayer

        # #layer three's stochastic resonator
        # #instantiate a new rng
        # #if we are explicitly forcing data dependence with the RNG processing
        # if not self.explicitDataDependence:
        #     self.resonators[3] = rngSource(standardDev)(previousLayerInput)
        #     #for now, the coupling weights will be constant (and learned not as a function of the particular input, but of the statistics of the input data and the neural network as it learns)        
        #     self.couplingLayers[3] = layers.Conv2D(256, (3,3), activation='linear', padding='same')(self.resonators[3])
        #     #compute output of the combination of the weighted RNG samples with the previous layer's output
        #     previousLayer = previousLayer + self.couplingLayers[3]
        #     previousAdvLayer = previousAdvLayer + self.couplingLayers[3]

        # else:
        #     self.resonators[3] = rngSource(standardDev)(previousLayer)
        #     #for now, the coupling weights will be constant (and learned not as a function of the particular input, but of the statistics of the input data and the neural network as it learns)        
        #     self.couplingLayers[3] = layers.Conv2D(256, (3,3), activation='linear', padding='same')(previousLayerInput)
        #     #compute output of the combination of the weighted RNG samples with the previous layer's output
        #     previousLayer = previousLayer + tf.keras.layers.Multiply()([self.couplingLayers[3], self.resonators[3]])
        #     previousAdvLayer = previousAdvLayer + tf.keras.layers.Multiply()([self.couplingLayers[3], self.resonators[3]])

        
        # #droput after noise
        # if (self.dropoutRate > 0):
        #     previousLayer = Dropout(dropout_rate)(previousLayer)
        #     previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        # pooling
        self.poolingLayers[3] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[3](previousLayer)
        previousAdvLayer = self.poolingLayers[3](previousAdvLayer)
        previousLayerInput = previousLayer

        #organize intermediate outputs
        self.hiddenModelOutputs[3] = previousLayer
        self.hiddenAdvModelOutputs[3] = previousAdvLayer

        #layer 4
        self.hiddenLayers[4] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[4] = self.hiddenLayers[4]
        previousLayer = self.hiddenLayers[4](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[4](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[4] = previousLayer
        self.hiddenAdvModelOutputs[4] = previousAdvLayer

        # #layer four's stochastic resonator
        # #instantiate a new rng
        # #if we are explicitly forcing data dependence with the RNG processing
        # if not self.explicitDataDependence:
        #     self.resonators[4] = rngSource(standardDev)(previousLayerInput)
        #     #for now, the coupling weights will be constant (and learned not as a function of the particular input, but of the statistics of the input data and the neural network as it learns)        
        #     self.couplingLayers[4] = layers.Conv2D(512, (3,3), activation='linear', padding='same')(self.resonators[4])
        #     #compute output of the combination of the weighted RNG samples with the previous layer's output
        #     previousLayer = previousLayer + self.couplingLayers[4]
        #     previousAdvLayer = previousAdvLayer + self.couplingLayers[4]

        # else:
        #     self.resonators[4] = rngSource(standardDev)(previousLayer)
        #     #for now, the coupling weights will be constant (and learned not as a function of the particular input, but of the statistics of the input data and the neural network as it learns)        
        #     self.couplingLayers[4] = layers.Conv2D(512, (3,3), activation='linear', padding='same')(previousLayerInput)
        #     #compute output of the combination of the weighted RNG samples with the previous layer's output
        #     previousLayer = previousLayer + tf.keras.layers.Multiply()([self.couplingLayers[4], self.resonators[4]])
        #     previousAdvLayer = previousAdvLayer + tf.keras.layers.Multiply()([self.couplingLayers[4], self.resonators[4]])
        # #droput after noise
        # if (self.dropoutRate > 0):
        #     previousLayer = Dropout(dropout_rate)(previousLayer)
        #     previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #organize intermediate outputs
        self.hiddenModelOutputs[4] = previousLayer
        self.hiddenAdvModelOutputs[4] = previousAdvLayer

        # pooling
        self.poolingLayers[4] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[4](previousLayer)
        previousAdvLayer = self.poolingLayers[4](previousAdvLayer)
        previousLayerInput = previousLayer

        #dense layers
        previousLayer = layers.Flatten()(previousLayer)
        previousAdvLayer = layers.Flatten()(previousAdvLayer)

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        self.penultimateDenseLayer = layers.Dense(1024, activation=self.chosenActivation, kernel_regularizer=regularizers.l2(5e-5))
        self.hiddenLayers[5] = self.penultimateDenseLayer
        self.hiddenAdvLayers[5] = self.penultimateDenseLayer
        previousLayer = self.penultimateDenseLayer(previousLayer)
        previousAdvLayer = self.penultimateDenseLayer(previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[5] = previousLayer
        self.hiddenAdvModelOutputs[5] = previousAdvLayer
        
        # #dense layer's stochastic resonator
        # if (self.stochasticResonatorOn):
        #     self.resonators[5] =  tf.keras.layers.GaussianNoise(1)(tf.zeros_like(self.hiddenModelOutputs[4]), training=True)#rngSource(std_dev=standardDev)(self.hiddenModelOutputs[4])#tf.cast(tf.random.normal(shape=tf.shape(self.hiddenModelOutputs[4]), mean=0, stddev=1), dtype=tf.float32)
            
        #     #for now, the coupling weights will be constant (and learned not as a function of the particular input, but of the statistics of the input data and the neural network as it learns)
        #     self.couplingLayers[5] = layers.Dense(128, activation='linear')(self.resonators[5])#tf.reshape(layers.Dense(int(t

            
        #     #compute output of the combination of the weighted RNG samples with the previous layer's output
        #     previousLayer = previousLayer + self.couplingLayers[5]
        #     previousAdvLayer = previousAdvLayer + self.couplingLayers[5]

        #organize intermediate outputs
        self.hiddenModelOutputs[5] = previousLayer
        self.hiddenAdvModelOutputs[5] = previousAdvLayer
        
        #droput after noise
        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #add the output layer
        self.outputLayer = layers.Dense(output_dimension, activation='softmax', name='outputLayer')#layers.Softmax()
        self.outputActivity = self.outputLayer(previousLayer)
        self.advOutputActivity = self.outputLayer(previousAdvLayer)

        #set up the logits layer (not just breaking apart the outputlayer because we want to be able to read in old pretrained models, so we'll just invert for this
        #softmax^-1 (X) at coordinate i = log(X_i) - log(\sum_j exp(X_j))
        self.penultimateLayer = self.logitsActivity = K.log(self.outputActivity) - K.log(K.sum(K.exp(self.outputActivity)))
        # self.advPenultimateLayer = self.advLogitsActivity = K.log(self.advOutputActivity) + K.log(K.sum(K.exp(self.advOutputActivity)))
        # # setup the models with which we can see states of hidden layers
        numberOfHiddenLayers = len(self.hiddenLayers)-1

        #collect adversarial projections and benign projections
        benProjs = layers.concatenate([layers.Flatten()(self.hiddenModelOutputs[curLayer]) for curLayer in range(numberOfHiddenLayers)])
        advProjs = layers.concatenate([layers.Flatten()(self.hiddenAdvModelOutputs[curLayer]) for curLayer in range(numberOfHiddenLayers)])
        self.benProjs, self.advProjs = benProjs, advProjs

        #define our custom loss function depending on how the intializer wants to regularize (i.e., the "reg" argument)
        #this is cross_entropy + \sum_layers(abs(benign_projection-adv_projection))
        self.usingAdvReg = usingAdvReg
        self.reg = reg
        if (not usingAdvReg):
            def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                def customLoss(y_true, y_pred):
                    return K.categorical_crossentropy(y_true, y_pred)
                return customLoss
        else:
            if (reg ==  'LDR'):
                def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.max(K.abs(benProjs - advProjs))/(tf.cast(K.max(K.abs(benProjs)), tf.float32))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))/(tf.cast(K.sum(K.square(benProjs)), tf.float32))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))/(tf.cast(K.sum(K.square(benProjs)), tf.float32))
                        return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(K.sum(K.abs(benProjs)), tf.float32))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(tf.shape(benProjs)[0], tf.float32))
                    return customLoss
            elif (reg ==  'FIM'):
                # dS = tf.gradients(self.outputLayer, self.inputLayer)
                # dS_2 = tf.matmul(dS, tf.reshape(dS, (dS.shape[1], dS.shape[0])))
                # eigs = tf.linalg.eigvals(dS_2)
                ps = tf.divide(tf.ones(shape=(tf.shape(self.outputActivity))), tf.where(self.outputActivity > 0.1, self.outputActivity, tf.ones_like(self.outputActivity)))
                def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(ps)
                    return customLoss

            
             
                    
        # optimization details
        # def lr_scheduler(epoch):
        #     return self.learning_rate * (0.5 ** (epoch // self.learning_rate_drop))
        # self.lr_scheduler = lr_scheduler
        # self.reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)
        if (optimizer==None):
            self.sgd = tf.keras.optimizers.Nadam()#Adadelta(learning_rate=self.learning_rate)
            self.reduceLR = None
        elif(optimizer=='SGD'):
            def lr_scheduler(epoch):
                return 0.1 * (0.5 ** (epoch // 20))#learning_rate * (0.5 ** (epoch // 20))

            self.reduceLR = keras.callbacks.LearningRateScheduler(lr_scheduler)
            self.sgd = tf.keras.optimizers.SGD(lr=.10, decay=1e-5, momentum=0.9, nesterov=True)
        # self.sgd = tf.keras.optimizers.SGD(lr=self.learning_rate, decay=0.000001, momentum=0.9, nesterov=True)

        #set up data augmentation
        self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)

        #convert self.hiddenAdvLayers to a list for the model compilation, ascending order of keys is order of layers
        #outputsList is a list of outputs of the model constructed so that the first entry is the true output (ie prediction) layer
        #and each subsequent (i, i+1)th entries are the pair of hiddenAdvLayer, hiddenBenignLayer activations
        #this is going to be useful for calculating the MAE between benignly and adversarially induced hidden states
        outputsList = [self.outputActivity]
        oneSideOutputsList = [self.outputActivity]
        for curHiddenLayer in range(len(self.hiddenAdvModelOutputs))[:-1]:
            oneSideOutputsList.append(self.hiddenModelOutputs[curHiddenLayer])
            outputsList.append(self.hiddenAdvModelOutputs[curHiddenLayer])
            outputsList.append(self.hiddenModelOutputs[curHiddenLayer])

        mainOutputList = [self.outputActivity]
        if (self.dualOutputs):
            mainOutputList.append(self.penultimateLayer)

        # instantiate and compile the model
        self.customLossWrapper = customLossWrapper
        self.singleInSingleOutModel = Model(inputs=[self.inputLayer], outputs=[self.outputActivity], name='sisoModel')
        self.model = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=mainOutputList, name='ddsr_vgg_16')
        #if we want to use this as a frozen model
        if (freezeWeights):
            for curWeights in range(len(self.model.layers)):
                self.model.layers[curWeights].trainable = False
        self.model.compile(loss=customLossWrapper(benProjs, advProjs, self.penaltyCoeff), metrics=['acc'], optimizer=self.sgd)
        #compile the siso model
        self.singleInSingleOutModel.compile(loss= tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'], optimizer=self.sgd)

        #setup the models with which we can see states of hidden layers
        self.hiddenModel = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=outputsList, name='hidden_ddsr_vgg_16')
        # self.hiddenOneSideModel = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=oneSideOutputsList, name='hidden_oneside_hlrr_vgg_16')
        # self.hiddenJointLatentModel = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=[benProjs], name='hiddenJointLatentModel')
        # self.logitModel = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=[self.logitsActivity], name='hiddenJointLatentModel')
        # double check weight trainability bug
        allVars = self.model.variables
        trainableVars = self.model.trainable_variables
        allVarNames = [self.model.variables[i].name for i in range(len(self.model.variables))]
        trainableVarNames = [self.model.trainable_variables[i].name for i in range(len(self.model.trainable_variables))]
        nonTrainableVars = np.setdiff1d(allVarNames, trainableVarNames)

        # for i, var in enumerate(self.model.variables):
        #     print(self.model.variables[i].name)
        # for i, var in enumerate(self.model.trainable_variables):
        #     print(self.model.trainable_variables[i].name)

        if (verbose):
            self.model.summary()
            if (len(nonTrainableVars) > 0):
                print('the following variables are set to non-trainable; ensure that this is correct before publishing!!!!')
            print(nonTrainableVars)

        #set data statistics to default values
        self.mean = 0
        self.stddev = 1

    #this method clears all models compiled by buildModel
    def deleteModels(self):
        try:
            del self.model
        except:
            pass
        
        try:
            del self.advInputActivity
        except:
            pass
        try:
            del self.advOutputActivity
        except:
            pass

        try:
            del self.advProjs
        except:
            pass

        try:
            del benProjs
        except:
            pass

        try:
            for curLayer in self.couplingLayers:
                del curLayer
        except:
            pass

        try:
            for curLayer in self.hiddenAdvLayers:
                try:
                    del curLayer
                except:
                    pass
        except:
            pass

        try:
            for curLayer in self.hiddenAdvModelOutputs:
                try:
                    del curLayer
                except:
                    pass
        except:
            pass

        try:
            for curLayer in self.hiddenLayers:
                try:
                    del curLayer
                except:
                    pass
        except:
            pass

        try:
            del self.inputLayer
        except:
            pass
        
        try:
            del self.logitsActivity
        except:
            pass

        try:
            del self.outputActivity
        except:
            pass                
        
        try:
            del self.outputLayer
        except:
            pass        
        try:
            del self.penultimateDenseLayer
        except:
            pass        
        try:
            del self.penultimateLayer
        except:
            pass        
        try:
            for curLayer in self.poolingLayers:
                try:
                    del curLayer
                except:
                    pass
        except:
            pass        
        try:
            for curRes in self.resonators:
                try:
                    del curRes
                except:
                    pass
        except:
            pass
        try:
           for callback in self.callbackList:
                del callback 
        except:
            pass

        try:
            del self.singleInSingleOutModel
        except:
            pass
        
        try:
            del self.hiddenModel
        except:
            pass
        
        try: 
            del self.hiddenOneSideModel
        except:
            pass
        
        try: 
            del self.hiddenJointLatentModel
        except:
            pass
        
        try: 
            del self.logitModel
        except:
            pass
        
        try:
            del self.sgd
        except:
            pass
        try:
            del self.generator
        except:
            pass
        gc.collect()
        tf.keras.backend.clear_session()
        return
    #this routine is used to collect statistics on training data, as well as to preprocess the training data by normalizing
    #i.e. centering and dividing by standard deviation
    def normalize(self, inputData, storeStats=False):
        if (storeStats):
            self.mean = np.mean(inputData)
            self.stddev = np.std(inputData)
            print("mean:%s; std:%s"%(str(self.mean), str(self.stddev)))
        outputData = (inputData-self.mean)/(self.stddev + 0.0000001)
        return outputData

    # routine to get a pointer to the optimizer of this model
    def getOptimizer(self):
        return self.sgd
    #
    def getVGGWeights(self):
        return self.model.get_weights().copy()

    def getParameterCount(self):
        return self.model.count_params()

    # handle data augmentation with multiple inputs (example found on https://stackoverflow.com/questions/49404993/keras-how-to-use-fit-generator-with-multiple-inputs
    #so thanks to loannis and Julian
    # this bit shamelessly stolen from geifmany on github, because we're starting with their benchmark
    # https: // github.com / geifmany / cifar - vgg / blob / master / cifar10vgg.py
    # also I didn't know we could do data augmentation in so few lines
    def multiInputDataGenerator(self, X1, X2, Y, batch_size):
        genX1 = self.generator.flow(X1, Y, batch_size=batch_size)
        genX2 = self.generator.flow(X2, Y, batch_size=batch_size)

        while True:
            X1g = genX1.next()
            X2g = genX2.next()
            yield [X1g[0], X2g[0]], X1g[1]

    #adversarial order parameter tells us if we're doing adversarial training, so we know if we should normalize to the first or second argument
    def train(self, inputTrainingData, trainingTargets, inputValidationData, validationTargets, training_epochs=1, 
                normed=False, monitor='val_loss', patience=10, testBetweenEpochs=None, 
                inputTestingData=None, inputTestingTargets=None, inputAdvPerturbations=None,
                inputTestingAGNPerturbations=None, model_path=None, keras_batch_size=64, 
                noisePowers=None, advPowers=None, dataAugmentation=False, adversarialOrder=0):
        #if a path isn't provided by caller, just use the current time for restoring best weights from fit
        if (model_path is None):
            model_path = os.path.join(tmpPathPrefix+'modelCheckpoints/', 'hlr_vgg16_'+str(int(round(time.time()*1000))))

        #if the data are not normalized, normalize them
        trainingData, validationData, testingData, testingAdvData, testingAGNData = [[],[]], [[],[]], [[],[]], [[],[]], [[],[]]
        if (not normed):
            #don't store stats from the adversarially attacked data
            if (adversarialOrder == 0):
                trainingData[0] = self.normalize(inputTrainingData[0], storeStats=True)
                trainingData[1] = self.normalize(inputTrainingData[1], storeStats=False)
            else:
                trainingData[1] = self.normalize(inputTrainingData[1], storeStats=True)
                trainingData[0] = self.normalize(inputTrainingData[0], storeStats=False)
            #also don't store stats from validation data
            validationData[0] = self.normalize(inputValidationData[0], storeStats=False)
            validationData[1] = self.normalize(inputValidationData[1], storeStats=False)

            if (testBetweenEpochs is not None and inputTestingData is not None and inputTestingTargets is not None):
                testingTargets = inputTestingTargets
                testingData = self.normalize(inputTestingData[0], storeStats=False)
                testingAdvData = inputAdvPerturbations
                testingAGNData = inputTestingAGNPerturbations
                # testingAdvData = self.normalize(inputAdvPerturbations, storeStats=False)
                # testingAGNData = self.normalize(inputTestingAGNPerturbations, storeStats=False)

        else:
            trainingData[0] = inputTrainingData[0]
            trainingData[1] = inputTrainingData[1]
            validationData[0] = inputValidationData[0]
            validationData[1] = inputValidationData[1]
            if (testBetweenEpochs is not None and inputTestingData is not None and inputTestingTargets is not None):
                testingTargets = inputTestingTargets
                testingData = inputTestingData
                testingAdvData = inputAdvPerturbations
                testingAGNData = inputTestingAGNPerturbations

        #handle data augmentation
        #this bit shamelessly stolen from geifmany on github, because we're starting with their benchmark
        #https: // github.com / geifmany / cifar - vgg / blob / master / cifar10vgg.py

        #collect our callbacks
        # earlyStopper = EarlyStopping(monitor=monitor, mode='min', patience=patience,
        #                              verbose=1, min_delta=defaultLossThreshold)
        checkpoint = ModelCheckpoint(model_path, verbose=1, monitor=monitor, save_weights_only=True,
                                     save_best_only=True, mode='auto')
        callbackList = [checkpoint]#[]#[earlyStopper, checkpoint]
        if (self.reduceLR is not None):
            callbackList.append(self.reduceLR)
        
        if (not dataAugmentation):
            # set up data augmentation
            self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                                samplewise_center=False,  # set each sample mean to 0
                                                featurewise_std_normalization=False,
                                                # divide inputs by std of the dataset
                                                samplewise_std_normalization=False,  # divide each input by its std
                                                zca_whitening=False,  # apply ZCA whitening
                                                # randomly shift images vertically (fraction of total height)
                                                horizontal_flip=False,# randomly flip images
                                                vertical_flip=False)
            self.generator.fit(trainingData[0])
        else:
            # set up data augmentation
            self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                                samplewise_center=False,  # set each sample mean to 0
                                                featurewise_std_normalization=False,
                                                # divide inputs by std of the dataset
                                                samplewise_std_normalization=False,  # divide each input by its std
                                                zca_whitening=False,  # apply ZCA whitening
                                                rotation_range=15,
                                                # randomly rotate images in the range (degrees, 0 to 180)
                                                width_shift_range=0.1,
                                                # randomly shift images horizontally (fraction of total width)
                                                height_shift_range=0.1,
                                                # randomly shift images vertically (fraction of total height)
                                                horizontal_flip=False,  # randomly flip images
                                                vertical_flip=False)
            self.generator.fit(trainingData[0])
        
        # tf.saved_model.save(self.model, 'modelForRoman')
        if (testBetweenEpochs is not None):
            numberOfAdvSamplesTest = testingAdvData.shape[0]
            evalBatchSize = np.min([numberOfAdvSamplesTest, keras_batch_size])
            history, benignResults, advResults, agnResults = [], [], [], []
            for curEpoch in range(training_epochs):
                print("current epoch: %s"%str(curEpoch))
                history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, batch_size=keras_batch_size),
                                        steps_per_epoch=trainingData[0].shape[0] // keras_batch_size,
                                        epochs=1, validation_data=(validationData, validationTargets),
                                        callbacks=callbackList, verbose=1) #self.reduce_lr                                    
                #if we're at the right interval between epochs and we're testing while training, do that here
                if ((testBetweenEpochs is not None) and (curEpoch % testBetweenEpochs == 0)):
                    benignEval = self.model.evaluate([testingData, testingData], testingTargets, batch_size=evalBatchSize)[1]
                    advAccuracy, noiseAccuracy = [benignEval], [benignEval]
                    
                    #evaluate adversarial test acc
                    for curNoisePower in advPowers:
                        #calculate new testingAttacks
                        testingAttacks = testingData + curNoisePower*inputAdvPerturbations
                        #get performances
                        curAcc = self.model.evaluate([testingAttacks, testingAttacks], testingTargets, batch_size=evalBatchSize)[1]
                        advAccuracy.append(curAcc)
                    #evaluate agn test acc
                    for curNoiseIndex in range(len(noisePowers)):
                        #calculate new testingAttacks
                        corruptedTestX = testingData + testingAGNData[curNoiseIndex][:numberOfAdvSamplesTest, :, :, :]
                        #get performances
                        agnEval = self.model.evaluate([corruptedTestX, corruptedTestX], testingTargets, batch_size=evalBatchSize)[1]
                        noiseAccuracy.append(agnEval)

                    
                    if benignResults:
                        benignResults.append(benignEval)
                        agnResults.append(noiseAccuracy)
                        advResults.append(advAccuracy)
                    else:
                        benignResults = [benignEval]
                        agnResults = [noiseAccuracy]
                        advResults = [advAccuracy]
            if (not np.isnan(history.history['loss']).any() and not np.isinf(history.history['loss']).any()):
                self.model.load_weights(model_path)
            loss, acc = history.history['loss'], history.history['val_acc']
            return loss, acc, model_path, benignResults, agnResults, advResults
        else:
            history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, keras_batch_size),
                                    steps_per_epoch=trainingData[0].shape[0] // keras_batch_size,
                                    epochs=training_epochs, validation_data=(validationData, validationTargets),
                                    callbacks=callbackList, verbose=1) #self.reduce_lr
            if (not np.isnan(history.history['loss']).any() and not np.isinf(history.history['loss']).any()):
                self.model.load_weights(model_path)                      
            loss, acc = history.history['loss'], history.history['val_acc']
            return loss, acc, model_path

    #identical to the training method above, bu instead of taking adversarial data as arguments, this function accepts a string specifying attack
    #method and performs the attacks on the model being trained online, as that model's weights are tuned
    def trainOnline(self, inputTrainingData, inputTrainingTargets, inputValidationData, validationTargets, training_epochs=1, #adversarialAttackFunction=create_adversarial_pattern,
                normed=False, monitor='val_loss', patience=10, testBetweenEpochs=None, onlineAttackType='FGSM', attackParams={'trainingEpsilon': 0.001},
                samplesPerEval = None, inputTestingData=None, inputTestingTargets=None, inputAdvPerturbations=None, exponentiallyDistributedEpsilons = None,
                inputTestingAGNPerturbations=None, model_path=None, keras_batch_size=64, noisePowers=None, advPowers=None, dataAugmentation=False, adversarialOrder=0, 
                trainOnAWGNPerts=False, awgnPertsOnly=False, noisyTrainingSigma=1./256., singleAttack=False):

        if samplesPerEval is None:
            samplesPerEval = inputTrainingData.shape[0]
        #delete the old generator
        if (self.generator is not None):
            del self.generator
            gc.collect()
            
        #define the attack function
        if (onlineAttackType == 'FGSM'):            
            
            #define our FGSM routine
            loss_object = tf.keras.losses.CategoricalCrossentropy()
            # @tf.function(input_signature=(tf.TensorSpec(shape=inputTrainingData.shape, dtype=tf.float32), tf.TensorSpec(shape=(inputTrainingTargets.shape[0],self.output_dimension), dtype=tf.float32)))
            def create_adversarial_pattern(inputImageTensor, inputLabel):
                with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                    tape.watch(inputImageTensor)
                    prediction = self.model([inputImageTensor, inputImageTensor])
                    loss = loss_object(inputLabel, prediction)

                # Get the gradients of the loss w.r.t to the input image.
                gradient = tape.gradient(loss, inputImageTensor)
                # Get the sign of the gradients to create the perturbation
                signed_grad = tf.sign(gradient)
                del gradient, loss, prediction, inputImageTensor, inputLabel, tape
                gc.collect()
                return signed_grad
            self.createAdversarialPattern = create_adversarial_pattern
        #instantiate results
        loss, acc, model_path, history = None, None, None, None

        #if a path isn't provided by caller, just use the current time for restoring best weights from fit
        if (model_path is None):
            model_path = os.path.join(tmpPathPrefix+'modelCheckpoints/', 'hlr_vgg16_'+str(int(round(time.time()*1000))))

        #if the data are not normalized, normalize them
        trainingData, validationData, testingData, testingAdvData, testingAGNData = [[],[]], [[],[]], [[],[]], [[],[]], [[],[]]
        if (not normed):
            #don't store stats from the adversarially attacked data
            trainingData[0] = self.normalize(inputTrainingData, storeStats=True)
            validationData[0] = self.normalize(inputValidationData[0], storeStats=False)

            if (testBetweenEpochs is not None and inputTestingData is not None and inputTestingTargets is not None):
                testingTargets = inputTestingTargets
                testingData = self.normalize(inputTestingData, storeStats=False)
                testingAdvData = inputAdvPerturbations
                testingAGNData = inputTestingAGNPerturbations

        else:
            trainingData[0] = inputTrainingData
            validationData = inputValidationData
            
            if (testBetweenEpochs is not None and inputTestingData is not None and inputTestingTargets is not None):
                testingTargets = inputTestingTargets
                testingData = inputTestingData
                testingAdvData = inputAdvPerturbations
                testingAGNData = inputTestingAGNPerturbations

        #handle data augmentation
        #this bit shamelessly stolen from geifmany on github, because we're starting with their benchmark
        #https: // github.com / geifmany / cifar - vgg / blob / master / cifar10vgg.py

        #collect our callbacks
        # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
        #                                      profile_batch='10, 15')

        earlyStopper = EarlyStopping(monitor=monitor, mode='min', patience=patience,
                                     verbose=1, min_delta=defaultLossThreshold)
        checkpoint = ModelCheckpoint(model_path, verbose=1, monitor=monitor, save_weights_only=True,
                                     save_best_only=True, mode='auto')
        self.callbackList = [earlyStopper, checkpoint] #, tb_callback]
        if (self.reduceLR is not None):
            self.callbackList.append(self.reduceLR)
        
        if (not dataAugmentation):
            # set up data augmentation
            self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                                samplewise_center=False,  # set each sample mean to 0
                                                featurewise_std_normalization=False,
                                                # divide inputs by std of the dataset
                                                samplewise_std_normalization=False,  # divide each input by its std
                                                zca_whitening=False,  # apply ZCA whitening
                                                # randomly shift images vertically (fraction of total height)
                                                horizontal_flip=False,# randomly flip images
                                                vertical_flip=False)
            self.generator.fit(trainingData[0])
        else:
            # set up data augmentation
            self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                                                samplewise_center=False,  # set each sample mean to 0
                                                featurewise_std_normalization=False,
                                                # divide inputs by std of the dataset
                                                samplewise_std_normalization=False,  # divide each input by its std
                                                zca_whitening=False,  # apply ZCA whitening
                                                rotation_range=15,
                                                # randomly rotate images in the range (degrees, 0 to 180)
                                                width_shift_range=0.1,
                                                # randomly shift images horizontally (fraction of total width)
                                                height_shift_range=0.1,
                                                # randomly shift images vertically (fraction of total height)
                                                horizontal_flip=False,  # randomly flip images
                                                vertical_flip=False)
            self.generator.fit(trainingData[0])
            
        # tf.saved_model.save(self.model, 'modelForRoman')
    
        numberOfAdvSamplesTest = testingAdvData.shape[0]
        evalBatchSize = np.min([inputTrainingData.shape[0], keras_batch_size])
        history, benignResults, advResults, agnResults = [], [], [], []
        for curEpoch in range(training_epochs):
            print("current epoch: %s"%str(curEpoch))

            #stack training data 
            if (awgnPertsOnly):
                curNoisyTrainingSamples = inputTrainingData + np.random.normal(0, scale=noisyTrainingSigma, size=inputTrainingData.shape)
                trainingData = [inputTrainingData, curNoisyTrainingSamples] if adversarialOrder == 0 else [curNoisyTrainingSamples, inputTrainingData]
                trainingTargets = inputTrainingTargets
                history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, batch_size=keras_batch_size),\
                        steps_per_epoch=trainingData[0].shape[0] // keras_batch_size, epochs=1, validation_data=(validationData, validationTargets), 
                        callbacks=self.callbackList, verbose=1) #self.reduce_lrs
                loss, acc = history.history['loss'], history.history['val_acc']
            else:
                if ((not singleAttack) or curEpoch == 0):
                    #if an online attack type is specified, perform the attack online and here
                    if (onlineAttackType == "FGSM"):
                        if (not singleAttack):
                            print("executing online attack; current epoch: %s"%str(curEpoch))
                        else:
                            print("executing batch attack on first iteration; current epoch: %s"%str(curEpoch))
                        

                        totalSamples = inputTrainingData.shape[0]
                        numberOfCompleteMinibatches = np.floor(totalSamples/samplesPerEval).astype(int)
                        trainingPerturbations = -1*np.ones_like(inputTrainingData)
                        XfgsmTr = -1*np.ones_like(inputTrainingData)
                        if (exponentiallyDistributedEpsilons is not None):
                            curTrainingEpsilons = np.random.gamma(shape=0.25, scale=attackParams['trainingEpsilon']/exponentiallyDistributedEpsilons, size=inputTrainingData.shape[0])
                        else:
                            curTrainingEpsilons = attackParams['trainingEpsilon']*np.ones(shape=(inputTrainingData.shape[0],))
                        
                        curMiniBatch = 0
                        #run the attacks in this distributed way because we run into memory constraints pretty quickly otherwise
                        for curMiniBatch in tqdm(range(numberOfCompleteMinibatches)):
                            curBenignSliceX = inputTrainingData[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval), :]
                            curBenignSliceY = inputTrainingTargets[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval)]
                            trainingPerturbations[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval)] = K.eval(self.createAdversarialPattern(tf.constant(curBenignSliceX, dtype=tf.float32), tf.constant(curBenignSliceY, dtype=tf.float32)))
                            gc.collect()
                            #I can't find a nicely readable way to (in a single line) multiply the slice of epsilons along every element of the perturbations
                            #so I'm doing it in a loop
                            for curAttack in range(samplesPerEval):
                                XfgsmTr[curMiniBatch*samplesPerEval+curAttack] = curBenignSliceX[curAttack] + curTrainingEpsilons[curMiniBatch*samplesPerEval+curAttack]*trainingPerturbations[curMiniBatch*samplesPerEval+curAttack]

                            # XfgsmTr[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval-1), :] = curBenignSliceX + np.multiply(trainingEpsilons[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval-1)]*np.ones_like(trainingPerturbations[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval-1), :]), trainingPerturbations[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval-1), :])
                        #compute the remainder (if it's greater than zero) of attacks
                        curMiniBatch+=1
                        samplesInFinalMiniBatch = np.mod(totalSamples, samplesPerEval)
                        if (samplesInFinalMiniBatch > 0 and numberOfCompleteMinibatches > 0):
                            curBenignSliceX = inputTrainingData[-1*samplesInFinalMiniBatch:]
                            curBenignSliceY = inputTrainingTargets[-1*samplesInFinalMiniBatch:]
                            trainingPerturbations[-1*samplesInFinalMiniBatch:] = K.eval(self.createAdversarialPattern(tf.constant(curBenignSliceX, dtype=tf.float32), tf.constant(curBenignSliceY, dtype=tf.float32)))

                            #I can't find a nicely readable way to (in a single line) multiply the slice of epsilons along every element of the perturbations
                            #so I'm doing it in a loop
                            for curAttack in range(samplesInFinalMiniBatch):
                                XfgsmTr[curMiniBatch*samplesPerEval+curAttack] = curBenignSliceX[curAttack] + curTrainingEpsilons[curMiniBatch*samplesPerEval+curAttack]*trainingPerturbations[curMiniBatch*samplesPerEval+curAttack]
                        
                        print("attack complete")

                    if (trainOnAWGNPerts):
                        curNoisyTrainingSamples = inputTrainingData + np.random.normal(0, scale=noisyTrainingSigma, size=inputTrainingData.shape)
                        trainingData = [inputTrainingData, curNoisyTrainingSamples] if adversarialOrder == 0 else [curNoisyTrainingSamples, inputTrainingData]
                        trainingTargets = inputTrainingTargets
                        history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, batch_size=keras_batch_size),\
                            steps_per_epoch=trainingData[0].shape[0] // keras_batch_size, epochs=1, validation_data=(validationData, validationTargets), 
                            callbacks=self.callbackList, verbose=1) #self.reduce_lr
                        del history
                        print(gc.collect())
                        # trainingData = [inputTrainingData, XfgsmTr] if adversarialOrder == 0 else [XfgsmTr, inputTrainingData]
                        # trainingTargets = inputTrainingTargets
                        # curAdvSide = np.concatenate((inputTrainingData, XfgsmTr), axis=0)
                        # curAdvSide = np.concatenate((curAdvSide, curNoisyTrainingSamples), axis=0)
                        # curBenSide = np.repeat(inputTrainingData, 3, axis=0)
                        # curBenSide = np.concatenate((curBenSide, curNoisyTrainingSamples), axis=0)
                        # curAdvSide = np.concatenate((curAdvSide, curNoisyTrainingSamples), axis=0)
                        # trainingData = [curBenSide, curAdvSide] if adversarialOrder == 0 else [curAdvSide, curBenSide]
                        #trainingTargets = np.repeat(inputTrainingTargets, 4, axis=0)
                            
                
                    trainingData = [inputTrainingData, XfgsmTr] if adversarialOrder == 0 else [XfgsmTr, inputTrainingData]
                    trainingTargets = inputTrainingTargets
                    history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, batch_size=keras_batch_size),\
                    steps_per_epoch=trainingData[0].shape[0] // keras_batch_size, epochs=1, validation_data=(validationData, validationTargets), 
                    callbacks=self.callbackList, verbose=1) #self.reduce_lr
                    loss, acc = history.history['loss'], history.history['val_acc']
                    # tf.keras.backend.clear_session()
                    del history
                    print(gc.collect())
                    
                else:
                    trainingData = [inputTrainingData, inputTrainingData]
                    trainingTargets = inputTrainingTargets
                    
                    history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, batch_size=keras_batch_size),\
                        steps_per_epoch=trainingData[0].shape[0] // keras_batch_size, epochs=1, validation_data=(validationData, validationTargets), 
                        callbacks=self.callbackList, verbose=1) #self.reduce_lr
                    loss, acc = history.history['loss'], history.history['val_acc']
                    del history
            # trainingData[1] = trainingData[0]
            
            gc.collect()
            #if we're at the right interval between epochs and we're testing while training, do that here
            if ((testBetweenEpochs is not None) and (curEpoch % testBetweenEpochs == 0)):
                benignEval = self.model.evaluate([testingData, testingData], testingTargets, batch_size=evalBatchSize)[1]
                advAccuracy, noiseAccuracy = [benignEval], [benignEval]
                
                #evaluate adversarial test acc
                for curNoisePower in advPowers:
                    #calculate new testingAttacks
                    testingAttacks = testingData + curNoisePower*inputAdvPerturbations
                    #get performances
                    curAcc = self.model.evaluate([testingAttacks, testingAttacks], testingTargets, batch_size=evalBatchSize)[1]
                    advAccuracy.append(curAcc)
                #evaluate agn test acc
                for curNoiseIndex in range(len(noisePowers)):
                    #calculate new testingAttacks
                    corruptedTestX = testingData + testingAGNData[curNoiseIndex][:numberOfAdvSamplesTest, :, :, :]
                    #get performances
                    agnEval = self.model.evaluate([corruptedTestX, corruptedTestX], testingTargets, batch_size=evalBatchSize)[1]
                    noiseAccuracy.append(agnEval)

                
                if benignResults:
                    benignResults.append(benignEval)
                    agnResults.append(noiseAccuracy)
                    advResults.append(advAccuracy)
                else:
                    benignResults = [benignEval]
                    agnResults = [noiseAccuracy]
                    advResults = [advAccuracy]
        if (not np.isnan(loss).any() and not np.isinf(loss).any()):
            self.model.load_weights(model_path)
        for callback in self.callbackList:
            del callback 

        tf.keras.backend.clear_session()
        #tf.compat.v1.reset_default_graph()
        print(gc.collect())
        # device = cuda.get_current_device()
        # device.synchronize()
        return loss, acc, model_path, benignResults, agnResults, advResults
        # else:
        #     history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, keras_batch_size),
        #                             steps_per_epoch=trainingData[0].shape[0] // keras_batch_size,
        #                             epochs=training_epochs, validation_data=(validationData, validationTargets),
        #                             callbacks=callbackList, verbose=1) #self.reduce_lr
        #     if (not np.isnan(history.history['loss']).any() and not np.isinf(history.history['loss']).any()):
        #         self.model.load_weights(model_path)                      
        #     loss, acc = history.history['loss'], history.history['val_acc']
        # 
        # return loss, acc, model_path
                

    def storeWeightsDistribution(self, pathToWeightsDistribution=None):
        weightsDistributionFigureList = []


        return weightsDistributionFigureList
        

    def evaluate(self, inputData, targets, batchSize=None):
        evalData = [self.normalize(inputData[0], storeStats=False), self.normalize(inputData[1], storeStats=False)]
        fullEval = self.model.evaluate(evalData, targets, batch_size=batchSize)
        return fullEval

    #this method stores our models to the disk at the specified path + name of component of the class + h5
    #and stores all other parameters in a pickle file
    def storeModelToDisk(self, pathToFile):
        # #set up the data to pickle
        pickleBox = {'chosenActivation':self.chosenActivation,
                     'scaleMean':self.mean,
                     'scaleSTD':self.stddev,
                     'reg':self.reg}

        #pickle the above data
        pickle.dump(pickleBox, open(pathToFile+'_pickle', 'wb'))

        self.model.save_weights(pathToFile)

    #this method stores a tensorflowV2 instance to the drive at the path specified by the argument
    #this model can be loaded using model = tf.saved_model.load(path_to_dir)
    def storeTensorflowV2(self, pathToFile):
        tf.saved_model.save(self.singleInSingleOutModel, pathToFile)
        return
    # this method reads our models from the disk at the specified path + name of component of the class + h5
    # and reads  all other parameters from a pickle file
    def readModelFromDisk(self, pathToFile):

        #rebuild the model
        self.buildModel(self.input_dimension, self.output_dimension, self.number_of_classes,
                        loss_threshold=self.loss_threshold, patience=self.patience, dropout_rate=self.dropoutRate, 
                        stochasticResonatorOn = self.stochasticResonatorOn, max_relu_bound=self.max_relu_bound, 
                        adv_penalty=self.advPenalty, unprotected=self.usingAdvReg,
                        reg=self.reg, verbose=verbose)
        # set the vgg weights
        self.model.load_weights(pathToFile)
        # #read in the picklebox
        pickleBox = pickle.load(open(pathToFile+'_pickle', 'rb'))
        # # self.bottleneckLayer = pickleBox['bottleneckLayer']
        # # self.hiddenEncodingLayer = pickleBox['hiddenEncodingLayer']
        # # self.inputLayer = pickleBox['inputLayer']
        self.reg = pickleBox['reg']
        self.chosenActivation = pickleBox['chosenActivation']
        self.mean, self.std = pickleBox['scaleMean'], pickleBox['scaleSTD']

#method to train online in a distributed way (hopefully this will result in tricking tensorflow into releasing all memory consumed by a DDSR's graph after training is complete, because for some reason, del obj is not )
# def trainOnlineDistributed(string: argsDictPath):


# # self, inputTrainingData, inputTrainingTargets, inputValidationData, validationTargets, training_epochs=1, 
# #                 normed=False, monitor='val_loss', patience=10, testBetweenEpochs=None, onlineAttackType='FGSM', attackParams={'trainingEpsilon': 0.001},
# #                 samplesPerEval = None, inputTestingData=None, inputTestingTargets=None, inputAdvPerturbations=None, exponentiallyDistributedEpsilons = False,
# #                 inputTestingAGNPerturbations=None, model_path=None, keras_batch_size=64, noisePowers=None, advPowers=None, dataAugmentation=False, adversarialOrder=0, 
# #                 trainOnAWGNPerts=False, awgnPertsOnly=False, noisyTrainingSigma=1./256.

#     #read in parameters from the pickled argsDict

#     #instantiate our ddsr

#     #run the training
#     if samplesPerEval is None:
#         samplesPerEval = inputTrainingData.shape[0]
#     #delete the old generator
#     if (self.generator is not None):
#         del self.generator
#         gc.collect()
        
#     #define the attack function
#     if (onlineAttackType == 'FGSM'):            
        
#         #define our FGSM routine
#         loss_object = tf.keras.losses.CategoricalCrossentropy()
#         # @tf.function(input_signature=(tf.TensorSpec(shape=inputTrainingData.shape, dtype=tf.float32), tf.TensorSpec(shape=(inputTrainingTargets.shape[0],self.output_dimension), dtype=tf.float32)))
#         def create_adversarial_pattern(inputImageTensor, inputLabel):
#             with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
#                 tape.watch(inputImageTensor)
#                 prediction = self.model([inputImageTensor, inputImageTensor])
#                 loss = loss_object(inputLabel, prediction)

#             # Get the gradients of the loss w.r.t to the input image.
#             gradient = tape.gradient(loss, inputImageTensor)
#             # Get the sign of the gradients to create the perturbation
#             signed_grad = tf.sign(gradient)
#             del gradient, loss, prediction, inputImageTensor, inputLabel, tape
#             gc.collect()
#             return signed_grad
#         self.createAdversarialPattern = create_adversarial_pattern
#     #instantiate results
#     loss, acc, model_path, history = None, None, None, None

#     #if a path isn't provided by caller, just use the current time for restoring best weights from fit
#     if (model_path is None):
#         model_path = os.path.join(tmpPathPrefix+'modelCheckpoints/', 'hlr_vgg16_'+str(int(round(time.time()*1000))))

#     #if the data are not normalized, normalize them
#     trainingData, validationData, testingData, testingAdvData, testingAGNData = [[],[]], [[],[]], [[],[]], [[],[]], [[],[]]
#     if (not normed):
#         #don't store stats from the adversarially attacked data
#         trainingData[0] = self.normalize(inputTrainingData, storeStats=True)
#         validationData[0] = self.normalize(inputValidationData[0], storeStats=False)

#         if (testBetweenEpochs is not None and inputTestingData is not None and inputTestingTargets is not None):
#             testingTargets = inputTestingTargets
#             testingData = self.normalize(inputTestingData, storeStats=False)
#             testingAdvData = inputAdvPerturbations
#             testingAGNData = inputTestingAGNPerturbations

#     else:
#         trainingData[0] = inputTrainingData
#         validationData = inputValidationData
        
#         if (testBetweenEpochs is not None and inputTestingData is not None and inputTestingTargets is not None):
#             testingTargets = inputTestingTargets
#             testingData = inputTestingData
#             testingAdvData = inputAdvPerturbations
#             testingAGNData = inputTestingAGNPerturbations

#     #handle data augmentation
#     #this bit shamelessly stolen from geifmany on github, because we're starting with their benchmark
#     #https: // github.com / geifmany / cifar - vgg / blob / master / cifar10vgg.py

#     #collect our callbacks
#     # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
#     #                                      profile_batch='10, 15')

#     earlyStopper = EarlyStopping(monitor=monitor, mode='min', patience=patience,
#                                     verbose=1, min_delta=defaultLossThreshold)
#     checkpoint = ModelCheckpoint(model_path, verbose=1, monitor=monitor, save_weights_only=True,
#                                     save_best_only=True, mode='auto')
#     self.callbackList = [earlyStopper, checkpoint] #, tb_callback]
#     if (self.reduceLR is not None):
#         self.callbackList.append(self.reduceLR)
    
#     if (not dataAugmentation):
#         # set up data augmentation
#         self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
#                                             samplewise_center=False,  # set each sample mean to 0
#                                             featurewise_std_normalization=False,
#                                             # divide inputs by std of the dataset
#                                             samplewise_std_normalization=False,  # divide each input by its std
#                                             zca_whitening=False,  # apply ZCA whitening
#                                             # randomly shift images vertically (fraction of total height)
#                                             horizontal_flip=False,# randomly flip images
#                                             vertical_flip=False)
#         self.generator.fit(trainingData[0])
#     else:
#         # set up data augmentation
#         self.generator = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
#                                             samplewise_center=False,  # set each sample mean to 0
#                                             featurewise_std_normalization=False,
#                                             # divide inputs by std of the dataset
#                                             samplewise_std_normalization=False,  # divide each input by its std
#                                             zca_whitening=False,  # apply ZCA whitening
#                                             rotation_range=15,
#                                             # randomly rotate images in the range (degrees, 0 to 180)
#                                             width_shift_range=0.1,
#                                             # randomly shift images horizontally (fraction of total width)
#                                             height_shift_range=0.1,
#                                             # randomly shift images vertically (fraction of total height)
#                                             horizontal_flip=False,  # randomly flip images
#                                             vertical_flip=False)
#         self.generator.fit(trainingData[0])
        
#     # tf.saved_model.save(self.model, 'modelForRoman')

#     numberOfAdvSamplesTest = testingAdvData.shape[0]
#     evalBatchSize = np.min([inputTrainingData.shape[0], keras_batch_size])
#     history, benignResults, advResults, agnResults = [], [], [], []
#     for curEpoch in range(training_epochs):
#         print("current epoch: %s"%str(curEpoch))

#         #stack training data 
#         if (awgnPertsOnly):
#             curNoisyTrainingSamples = inputTrainingData + np.random.normal(0, scale=noisyTrainingSigma, size=inputTrainingData.shape)
#             trainingData = [inputTrainingData, curNoisyTrainingSamples] if adversarialOrder == 0 else [curNoisyTrainingSamples, inputTrainingData]
#             trainingTargets = inputTrainingTargets
#             history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, batch_size=keras_batch_size),\
#                     steps_per_epoch=trainingData[0].shape[0] // keras_batch_size, epochs=1, validation_data=(validationData, validationTargets), 
#                     callbacks=self.callbackList, verbose=1) #self.reduce_lrs
#             loss, acc = history.history['loss'], history.history['val_acc']
#         else:
#             #if an online attack type is specified, perform the attack online and here
#             if (onlineAttackType == "FGSM"):
#                 print("executing online attack; current epoch: %s"%str(curEpoch))
                

#                 totalSamples = inputTrainingData.shape[0]
#                 numberOfCompleteMinibatches = np.floor(totalSamples/samplesPerEval).astype(int)
#                 trainingPerturbations = -1*np.ones_like(inputTrainingData)
#                 XfgsmTr = -1*np.ones_like(inputTrainingData)
#                 if (exponentiallyDistributedEpsilons):
#                     curTrainingEpsilons = np.random.gamma(shape=0.25, scale=attackParams['trainingEpsilon'], size=inputTrainingData.shape[0])
#                 else:
#                     curTrainingEpsilons = attackParams['trainingEpsilon']*np.ones(shape=(inputTrainingData.shape[0],))
                
#                 curMiniBatch = 0
#                 #run the attacks in this distributed way because we run into memory constraints pretty quickly otherwise
#                 for curMiniBatch in tqdm(range(numberOfCompleteMinibatches)):
#                     curBenignSliceX = inputTrainingData[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval), :]
#                     curBenignSliceY = inputTrainingTargets[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval)]
#                     trainingPerturbations[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval)] = K.eval(self.createAdversarialPattern(tf.constant(curBenignSliceX, dtype=tf.float32), tf.constant(curBenignSliceY, dtype=tf.float32)))
#                     gc.collect()
#                     #I can't find a nicely readable way to (in a single line) multiply the slice of epsilons along every element of the perturbations
#                     #so I'm doing it in a loop
#                     for curAttack in range(samplesPerEval):
#                         XfgsmTr[curMiniBatch*samplesPerEval+curAttack] = curBenignSliceX[curAttack] + curTrainingEpsilons[curMiniBatch*samplesPerEval+curAttack]*trainingPerturbations[curMiniBatch*samplesPerEval+curAttack]

#                     # XfgsmTr[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval-1), :] = curBenignSliceX + np.multiply(trainingEpsilons[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval-1)]*np.ones_like(trainingPerturbations[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval-1), :]), trainingPerturbations[(curMiniBatch*samplesPerEval):((curMiniBatch+1)*samplesPerEval-1), :])
#                 #compute the remainder (if it's greater than zero) of attacks
#                 curMiniBatch+=1
#                 samplesInFinalMiniBatch = np.mod(totalSamples, samplesPerEval)
#                 if (samplesInFinalMiniBatch > 0 and numberOfCompleteMinibatches > 0):
#                     curBenignSliceX = inputTrainingData[-1*samplesInFinalMiniBatch:]
#                     curBenignSliceY = inputTrainingTargets[-1*samplesInFinalMiniBatch:]
#                     trainingPerturbations[-1*samplesInFinalMiniBatch:] = K.eval(self.createAdversarialPattern(tf.constant(curBenignSliceX, dtype=tf.float32), tf.constant(curBenignSliceY, dtype=tf.float32)))

#                     #I can't find a nicely readable way to (in a single line) multiply the slice of epsilons along every element of the perturbations
#                     #so I'm doing it in a loop
#                     for curAttack in range(samplesInFinalMiniBatch):
#                         XfgsmTr[curMiniBatch*samplesPerEval+curAttack] = curBenignSliceX[curAttack] + curTrainingEpsilons[curMiniBatch*samplesPerEval+curAttack]*trainingPerturbations[curMiniBatch*samplesPerEval+curAttack]
                
#                 print("attack complete")



#                 if (trainOnAWGNPerts):
#                     curNoisyTrainingSamples = inputTrainingData + np.random.normal(0, scale=noisyTrainingSigma, size=inputTrainingData.shape)
#                     trainingData = [inputTrainingData, curNoisyTrainingSamples] if adversarialOrder == 0 else [curNoisyTrainingSamples, inputTrainingData]
#                     trainingTargets = inputTrainingTargets
#                     history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, batch_size=keras_batch_size),\
#                         steps_per_epoch=trainingData[0].shape[0] // keras_batch_size, epochs=1, validation_data=(validationData, validationTargets), 
#                         callbacks=self.callbackList, verbose=1) #self.reduce_lr
#                     del history
#                     print(gc.collect())
#                     # trainingData = [inputTrainingData, XfgsmTr] if adversarialOrder == 0 else [XfgsmTr, inputTrainingData]
#                     # trainingTargets = inputTrainingTargets
#                     # curAdvSide = np.concatenate((inputTrainingData, XfgsmTr), axis=0)
#                     # curAdvSide = np.concatenate((curAdvSide, curNoisyTrainingSamples), axis=0)
#                     # curBenSide = np.repeat(inputTrainingData, 3, axis=0)
#                     # curBenSide = np.concatenate((curBenSide, curNoisyTrainingSamples), axis=0)
#                     # curAdvSide = np.concatenate((curAdvSide, curNoisyTrainingSamples), axis=0)
#                     # trainingData = [curBenSide, curAdvSide] if adversarialOrder == 0 else [curAdvSide, curBenSide]
#                     #trainingTargets = np.repeat(inputTrainingTargets, 4, axis=0)
                        
            
#                 trainingData = [inputTrainingData, XfgsmTr] if adversarialOrder == 0 else [XfgsmTr, inputTrainingData]
#                 trainingTargets = inputTrainingTargets
#                 history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, batch_size=keras_batch_size),\
#                 steps_per_epoch=trainingData[0].shape[0] // keras_batch_size, epochs=1, validation_data=(validationData, validationTargets), 
#                 callbacks=self.callbackList, verbose=1) #self.reduce_lr
#                 loss, acc = history.history['loss'], history.history['val_acc']
#                 # tf.keras.backend.clear_session()
#                 del history
#                 print(gc.collect())
                
#             else:
#                 trainingData = [inputTrainingData, inputTrainingData]
#                 trainingTargets = inputTrainingTargets
                
#                 history = self.model.fit(self.multiInputDataGenerator(trainingData[0], trainingData[1], trainingTargets, batch_size=keras_batch_size),\
#                     steps_per_epoch=trainingData[0].shape[0] // keras_batch_size, epochs=1, validation_data=(validationData, validationTargets), 
#                     callbacks=self.callbackList, verbose=1) #self.reduce_lr
#                 loss, acc = history.history['loss'], history.history['val_acc']
#                 del history
#             del XfgsmTr, trainingPerturbations,
#             gc.collect()
#         # trainingData[1] = trainingData[0]
        
#         gc.collect()
#         #if we're at the right interval between epochs and we're testing while training, do that here
#         if ((testBetweenEpochs is not None) and (curEpoch % testBetweenEpochs == 0)):
#             benignEval = self.model.evaluate([testingData, testingData], testingTargets, batch_size=evalBatchSize)[1]
#             advAccuracy, noiseAccuracy = [benignEval], [benignEval]
            
#             #evaluate adversarial test acc
#             for curNoisePower in advPowers:
#                 #calculate new testingAttacks
#                 testingAttacks = testingData + curNoisePower*inputAdvPerturbations
#                 #get performances
#                 curAcc = self.model.evaluate([testingAttacks, testingAttacks], testingTargets, batch_size=evalBatchSize)[1]
#                 advAccuracy.append(curAcc)
#             #evaluate agn test acc
#             for curNoiseIndex in range(len(noisePowers)):
#                 #calculate new testingAttacks
#                 corruptedTestX = testingData + testingAGNData[curNoiseIndex][:numberOfAdvSamplesTest, :, :, :]
#                 #get performances
#                 agnEval = self.model.evaluate([corruptedTestX, corruptedTestX], testingTargets, batch_size=evalBatchSize)[1]
#                 noiseAccuracy.append(agnEval)

            
#             if benignResults:
#                 benignResults.append(benignEval)
#                 agnResults.append(noiseAccuracy)
#                 advResults.append(advAccuracy)
#             else:
#                 benignResults = [benignEval]
#                 agnResults = [noiseAccuracy]
#                 advResults = [advAccuracy]
#     if (not np.isnan(loss).any() and not np.isinf(loss).any()):
#         self.model.load_weights(model_path)
#     for callback in self.callbackList:
#         del callback 

#     tf.keras.backend.clear_session()
#     #tf.compat.v1.reset_default_graph()
#     print(gc.collect())
#     # device = cuda.get_current_device()
#     # device.synchronize()
    
    
#     pickle these loss, acc, model_path, benignResults, agnResults, advResults

#     return the path of the file each of these are located in and the paths of the stored models


#the main body of this class pretrains a DDSR network
if __name__ == "__main__":
    #define test parameters
    # define parameters
    verbose = True
    defaultPatience = 10
    kerasBatchSize = 512 
    explicitDataDependence = True
    testFraction = 0.25
    numberOfClasses = 10
    # trainingSetSize = 100000
    numberOfAdvSamples = 1000
    trainingEpochs = 2048
    maxCarliniIts = 10000
    powers = [0.05, 0.1, 0.25, 0.5, 1, 2]

    # set up data
    # input image dimensions
    img_rows, img_cols = 32, 32

    inputDimension = (32, 32, 3)
    # read in cifar data
    # split data between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # scale and standardize (with respect to stats calculated on training set)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.
    # # for cifar mean: 0.4733649, std: 0.25156906
    # x_train = (x_train - 0.4733649) / (0.25156906)
    # x_test = (x_test - 0.4733649) / (0.25156906)

    #normalize the data once after the pretraining experiment
    mean = np.mean(x_train)
    std = np.std(x_train)
    # print(mean)
    # print(std)
    x_train -= mean
    x_test -= mean
    x_train /= std
    x_test /= std
    # print(x_test.shape)
    # print(x_train.shape)

    #split validation data
    valFrac = 0.1
    trainX, valX, trainY, valY = model_selection.train_test_split(x_train, y_train, shuffle=True,
                                                                    test_size=valFrac, random_state=42)

    print('beginning test')
    output_dimension = numberOfClasses
    ourModel = DDSR(inputDimension, output_dimension, number_of_classes=numberOfClasses, adv_penalty=0.0025,
                       loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0, explicitDataDependence=True,
                       max_relu_bound=1.1, reg='LDR', verbose=True, usingAdvReg=True, stochasticResonatorOn=False)

    ourModel.train([trainX, trainX], tf.keras.utils.to_categorical(trainY, numberOfClasses), [trainX, trainX], \
                   tf.keras.utils.to_categorical(trainY, numberOfClasses), training_epochs=trainingEpochs, normed=True,\
                   monitor='val_loss', keras_batch_size=kerasBatchSize, dataAugmentation=True, adversarialOrder=0)

    ourModel.model.save_weights('LDR_resonators_off_2048')
    # #     training_epochs=trainingEpochs, normed=False, monitor='val_loss', keras_batch_size=kerasBatchSize, dataAugmentation=True, adversarialOrder=0)
    
    # ourModel.buildModel(input_dimension, output_dimension, number_of_classes=numberOfClasses, optimizer=None, dual_outputs=False,
    #     loss_threshold=defaultLossThreshold, patience=10, dropout_rate=defaultDropoutRate, 
    #     max_relu_bound=None, adv_penalty=0.0025, adv_penalty_l1=0.0025, explicitDataDependence=True, usingAdvReg=True, reg='LDR', 
    #     stochasticResonatorOn=False, freezeWeights=False, verbose=False)
    # ourModel.model.load_weights('explicitDataDependenceUnprotectedDDSR_2048_resonatorsOff_bs512_5_17.h5')
    # ourModel.model.predict([np.expand_dims(np.zeros(shape=inputDimension), axis=0), np.expand_dims(np.zeros(shape=inputDimension), axis=0)])
    # ourModel.singleInSingleOutModel.predict(np.expand_dims(np.zeros(shape=inputDimension), axis=0))
    # ourModel.singleInSingleOutModel.compile(loss= tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'], optimizer=ourModel.sgd)
    # tf.saved_model.save(ourModel.singleInSingleOutModel, "explicitDataDependenceUnprotectedDDSR_2048_resonatorsOff_bs512_5_17")
    # # ourModel.storeTensorflowV2("v2Model_explicitDataDependenceUnprotectedDDSR_2048_bs_512_resonatorsOff.h5")
    # # unprotectedModel = DDSR(inputDimension, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0.05,
    # #                    loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0,#0.33,
    # #                    max_relu_bound=1.1, verbose=False, usingAdvReg=False, explicitDataDependence=explicitDataDependence, stochasticResonatorOn=True)
                    
    
    # unprotectedModel.model.summary()
    # # unprotectedModel.train([trainX, trainX], tf.keras.utils.to_categorical(trainY, numberOfClasses), [trainX, trainX], tf.keras.utils.to_categorical(trainY, numberOfClasses), \
    # #     training_epochs=trainingEpochs, normed=False, monitor='val_loss', keras_batch_size=kerasBatchSize, dataAugmentation=True, adversarialOrder=0)
    # unprotectedModel.trainOnline(trainX, tf.keras.utils.to_categorical(trainY, numberOfClasses), [trainX, trainX], tf.keras.utils.to_categorical(trainY, numberOfClasses), \
    #     training_epochs=trainingEpochs, normed=False, monitor='val_loss', keras_batch_size=kerasBatchSize, dataAugmentation=True, adversarialOrder=0)
    # unprotectedModel.model.save_weights('explicitDataDependenceUnprotectedDDSR.h5')
    
    # #test the unprotected model and report the results
    # testResults = unprotectedModel.evaluate([testX, testX], tf.keras.utils.to_categorical(testY, numberOfClasses), batchSize=1024)
    # print(testResults)

    # ourModel.model.save('hlrr_complete')
    # ourModel.hiddenModel.save('hlrr_complete_hidden')
    # ourModel.storeModelToDisk('hlrr_vgg_16_demo')
    # ourModel.readModelFromDisk('hlrr_vgg_16_demo')
    # print(ourModel.evaluate([np.expand_dims(testX, axis=3), np.expand_dims(np.zeros(testX.shape), axis=3)], tf.keras.utils.to_categorical(testY, numberOfClasses)))

    # # attack this vgg to generate samples with fgsm (#todo: repeat this with more expensive attacks)
    # # define fgsm method (thanks to TF at https://www.tensorflow.org/tutorials/generative/adversarial_fgsm)
    # loss_object = tf.keras.losses.CategoricalCrossentropy()
    
    # def create_adversarial_pattern(input_image, input_label):
    #     inputImageTensor = tf.cast(input_image, tf.float32)
    #     with tf.GradientTape() as tape:
    #         tape.watch(inputImageTensor)
    #         prediction = unprotectedModel.model([inputImageTensor, inputImageTensor])
    #         loss = loss_object(input_label, prediction)

    #     # Get the gradients of the loss w.r.t to the input image.
    #     gradient = tape.gradient(loss, inputImageTensor)
    #     # Get the sign of the gradients to create the perturbation
    #     signed_grad = tf.sign(gradient)
    #     return signed_grad


    # attackFunction = tf.function()
    # # set up containers to store perturbations
    # fgsmData = dict()
    # fgsmData['train'] = np.zeros((numberOfAdvSamples, 32, 32, 1))
    # fgsmData['test'] = np.zeros((numberOfAdvSamples, 32, 32, 1))
    # # attack the unprotected model
    # fgsmData['train'] = create_adversarial_pattern(trainX[:numberOfAdvSamples, :, :],
    #                                                tf.keras.utils.to_categorical(trainY[:numberOfAdvSamples],
    #                                                                              num_classes=numberOfClasses))
    # # fgsmData['test'] = create_adversarial_pattern(testX[:numberOfAdvSamples, :, :, :],
    # #                                               tf.keras.utils.to_categorical(testY[:numberOfAdvSamples],
    # #                                                                             num_classes=numberOfClasses))

    # print("attack complete")

    # #mock up the MAE vs layer plot for varying noise levels
    # #first collect the perturbed MAE
    # noiseLevels = [0.5, 1, 5, 10]
    # averageLayerwiseMAEs = dict()
    # for curNoiseLevel in noiseLevels:
    #     ourModel.train([trainX, trainX],
    #                    tf.keras.utils.to_categorical(trainY, numberOfClasses), 
    #                    [trainX, trainX], tf.keras.utils.to_categorical(trainY, numberOfClasses), training_epochs=trainingEpochs,
    #                    monitor='val_loss', patience=25, model_path=None, keras_batch_size=64,
    #                    dataAugmentation=True)
    #     trainingAttacks =trainX[:numberOfAdvSamples, :, :] + curNoiseLevel * K.eval(fgsmData['train'])
    #     testingAttacks = testX[:numberOfAdvSamples, :, :] + curNoiseLevel * K.eval(fgsmData['test'])
    #     curAdversarialPreds = ourModel.hiddenModel.predict([trainingAttacks,
    #                                                         np.expand_dims(trainX[:numberOfAdvSamples, :, :], axis=3)])
    #     curLayerWiseMAEs = []
    #     for curLayer in list(range(len(curAdversarialPreds)))[1::2]:
    #         curLayerAEs = np.abs(curAdversarialPreds[curLayer]-curAdversarialPreds[curLayer+1])
    #         curAdversarialMAEs = []
    #         init = time.time()
    #         a = np.sum(np.sum(np.sum(
    #             np.sum(np.abs(curAdversarialPreds[curLayer] - curAdversarialPreds[curLayer + 1]), axis=0) /
    #             curAdversarialPreds[curLayer].shape[0], axis=0) / curAdversarialPreds[curLayer].shape[1], axis=0) /
    #                    curAdversarialPreds[curLayer].shape[2], axis=0) / curAdversarialPreds[curLayer].shape[3]
    #         atime = time.time() - init
    #         curLayerWiseMAEs.append(a)
    #         # print((a, atime))
    #         init = time.time()
    #         b = np.mean(np.mean(
    #             np.mean(np.mean(np.abs(curAdversarialPreds[curLayer] - curAdversarialPreds[curLayer + 1]), axis=0), axis=0),
    #             axis=0), axis=0)
    #         btime = time.time() - init
    #         print((curLayer+1)/2)
    #         print((a, b))
    #         print((atime, btime))
    #     averageLayerwiseMAEs[curNoiseLevel] = curLayerWiseMAEs


    # #plot the maes
    # plt.figure()
    # numberOfLayers = np.round((len(curAdversarialPreds)-1)/2).astype(int)
    # for curLevel in noiseLevels:
    #     plt.plot(range(numberOfLayers), averageLayerwiseMAEs[curLevel], label='sigma = %s'%str(curLevel))

    # plt.legend()
    # plt.savefig('layerwiseMAEs.png')
    # plt.show()

# # restrict number of classes
#     trainXs, trainTargets = [], []
#     testXs, testTargets = [], []

#     for t in range(numberOfClasses):
#         curClassIndicesTraining = np.where(y_train == t)[0]
#         curClassIndicesTesting = np.where(y_test == t)[0]
#         if (trainingSetSize == -1):
#             # arrange training data
#             if (dataset == 'fashion_mnist'):
#                 curXTrain = np.expand_dims(x_train[curClassIndicesTraining, :, :], axis=3)
#                 curXTest = np.expand_dims(x_test[curClassIndicesTesting, :, :], axis=3)
#             else:
#                 curXTrain = x_train[curClassIndicesTraining, :, :]
#                 curXTest = x_test[curClassIndicesTesting, :, :]
#             # arrange testing data
#             # curXTest = np.squeeze(x_test[curClassIndicesTesting, :])

#         else:
#             if (dataset == 'fashion_mnist'):
#                 # arrange training data
#                 curXTrain = np.expand_dims(x_train[curClassIndicesTraining[:trainingSetSize], :, :], axis=3)
#                 # arrange testing data
#                 curXTest = np.expand_dims(x_test[curClassIndicesTesting[:trainingSetSize], :, :], axis=3)
#             else:
#                 # arrange training data
#                 curXTrain = x_train[curClassIndicesTraining[:trainingSetSize], :, :]
#                 # arrange testing data
#                 curXTest = x_test[curClassIndicesTesting[:trainingSetSize], :, :]

#         trainXs.append(curXTrain)
#         trainTargets.append((t * np.ones([curXTrain.shape[0], 1])))

#         testXs.append(curXTest)
#         testTargets.append(t * np.ones([curXTest.shape[0], 1]))

#     # stack our data
#     t = 0
#     stackedData, stackedTargets = np.array([]), np.array([])
#     for t in range(numberOfClasses):
#         if (verbose):
#             print('current class count')
#             print(t + 1)
#         stackedData = np.concatenate((stackedData, trainXs[t]), axis=0) if stackedData.size > 0 else trainXs[t]
#         stackedData = np.concatenate((stackedData, testXs[t]), axis=0)
#         stackedTargets = np.concatenate((stackedTargets, trainTargets[t]), axis=0) if stackedTargets.size > 0 else \
#         trainTargets[t]
#         stackedTargets = np.concatenate((stackedTargets, testTargets[t]), axis=0)

#     #here, in pre-training, we only want to consider our trainingData, and the test data will be used int he adv robustness experiment
#     trainX, testX, trainY, testY = model_selection.train_test_split(stackedData, stackedTargets, shuffle=True,
                                                                    # test_size=testFraction, random_state=42)