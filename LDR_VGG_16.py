########################################################################################################################
########################################################################################################################
# Authors: David Schwartz
#
#
########################################################################################################################
import sys
# if ('linux' not in sys.platform):
#     import pyBILT_common as pyBILT

# import infotheory
# import lempel_ziv_complexity
# import scipy
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
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, TimeDistributed, Conv1D, BatchNormalization

# tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
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
# from ops import *
import itertools

# from dit.other import renyi_entropy

# bucketBoundaries = np.logspace(start=0, stop=1.1, num=1024)
numberOfBins = 8
@tf.function
def lempel_ziv_complexity(sequence):
    r""" Manual implementation of the Lempel-Ziv complexity.

    It is defined as the number of different substrings encountered as the stream is viewed from begining to the end.
    As an example:

    >>> s = '1001111011000010'
    >>> lempel_ziv_complexity(s)  # 1 / 0 / 01 / 11 / 10 / 110 / 00 / 010
    8
    """
    sub_strings = list()
    n = tf.size(sequence)

    ind = 0
    inc = 1
    while ind + inc <= n:
        # if ind + inc > n:
        #     break
        sub_str = sequence[ind : ind + inc]
        isIn = False
        for curSubString in sub_strings:
            if sub_str == curSubString:
                isIn = True
                break
        if isIn:
            inc += 1
        else:
            sub_strings.append(sub_str)
            ind += inc
            inc = 1
    return len(sub_strings)/(tf.cast(n, tf.float32)/K.log(tf.cast(n, tf.float32))/K.log(tf.cast(numberOfBins, tf.float32)  ))
########################################################################################################################

defaultPatience = 25
defaultLossThreshold = 0.00001
defaultDropoutRate = 0

class LDRVGG(object):
    def __init__(self, input_dimension, output_dimension, number_of_classes=2, optimizer=None, dual_outputs=False,
        loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=defaultDropoutRate, reg='HLDR',
        max_relu_bound=None, adv_penalty=0.01, adv_penalty_l1=0.01, unprotected=False, freezeWeights=False, verbose=False):

        self.buildModel(input_dimension, output_dimension, number_of_classes=number_of_classes, dual_outputs=dual_outputs,
                        loss_threshold=loss_threshold, patience=patience, dropout_rate=dropout_rate,
                        max_relu_bound=max_relu_bound, adv_penalty=adv_penalty, adv_penalty_l1=adv_penalty_l1, unprotected=unprotected,
                        optimizer=optimizer, reg=reg, freezeWeights=freezeWeights, verbose=verbose)


    def buildModel(self, input_dimension, output_dimension, number_of_classes=2, optimizer=None, dual_outputs=False,
        loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=defaultDropoutRate, 
        max_relu_bound=None, adv_penalty=0.01, adv_penalty_l1=0.01, unprotected=False, reg='HLDR', freezeWeights=False, verbose=False):
        self.input_dimension, self.output_dimension = input_dimension, np.copy(output_dimension)
        self.advPenalty = np.copy(adv_penalty)

        self.loss_threshold, self.number_of_classes = np.copy(loss_threshold), np.copy(number_of_classes)
        self.dropoutRate, self.max_relu_bound = np.copy(dropout_rate), np.copy(max_relu_bound)
        self.patience = np.copy(patience)
        self.dualOutputs = dual_outputs
        ##self.learning_rate, self.learning_rate_drop = np.copy(learning_rate), np.copy(learning_rate_drop)
        self.image_size = 32
        self.num_channels = 3
        self.num_labels = np.copy(number_of_classes)
        self.penaltyCoeff = np.copy(adv_penalty)
        self.penaltyCoeffL1 = np.copy(adv_penalty_l1)

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

        #define hidden layer outputs
        self.hiddenModelOutputs, self.hiddenAdvModelOutputs = dict(), dict()

        #layer 0
        self.hiddenLayers[0] = layers.Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[0] = self.hiddenLayers[0]
        previousLayer = self.hiddenLayers[0](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[0](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[0] = previousLayer
        self.hiddenAdvModelOutputs[0] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 1
        self.hiddenLayers[1] = layers.Conv2D(64, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[1] = self.hiddenLayers[1]
        previousLayer = self.hiddenLayers[1](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[1](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[1] = previousLayer
        self.hiddenAdvModelOutputs[1] = previousAdvLayer

        #pooling
        self.poolingLayers[0] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[0](previousLayer)
        previousAdvLayer = self.poolingLayers[0](previousAdvLayer)

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 2
        self.hiddenLayers[2] = layers.Conv2D(128, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[2] = self.hiddenLayers[2]
        previousLayer = self.hiddenLayers[2](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[2](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[2] = previousLayer
        self.hiddenAdvModelOutputs[2] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 3
        self.hiddenLayers[3] = layers.Conv2D(128, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[3] = self.hiddenLayers[3]
        previousLayer = self.hiddenLayers[3](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[3](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[3] = previousLayer
        self.hiddenAdvModelOutputs[3] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        # pooling
        self.poolingLayers[1] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[1](previousLayer)
        previousAdvLayer = self.poolingLayers[1](previousAdvLayer)


        #layer 4
        self.hiddenLayers[4] = layers.Conv2D(256, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[4] = self.hiddenLayers[4]
        previousLayer = self.hiddenLayers[4](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[4](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[4] = previousLayer
        self.hiddenAdvModelOutputs[4] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 5
        self.hiddenLayers[5] = layers.Conv2D(256, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[5] = self.hiddenLayers[5]
        previousLayer = self.hiddenLayers[5](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[5](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[5] = previousLayer
        self.hiddenAdvModelOutputs[5] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 6
        self.hiddenLayers[6] = layers.Conv2D(256, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[6] = self.hiddenLayers[6]
        previousLayer = self.hiddenLayers[6](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[6](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[6] = previousLayer
        self.hiddenAdvModelOutputs[6] = previousAdvLayer

        # pooling
        self.poolingLayers[2] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[2](previousLayer)
        previousAdvLayer = self.poolingLayers[2](previousAdvLayer)

        #layer 7
        self.hiddenLayers[7] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[7] = self.hiddenLayers[7]
        previousLayer = self.hiddenLayers[7](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[7](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[7] = previousLayer
        self.hiddenAdvModelOutputs[7] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 8
        self.hiddenLayers[8] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[8] = self.hiddenLayers[8]
        previousLayer = self.hiddenLayers[8](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[8](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[8] = previousLayer
        self.hiddenAdvModelOutputs[8] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 9
        self.hiddenLayers[9] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[9] = self.hiddenLayers[9]
        previousLayer = self.hiddenLayers[9](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[9](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[9] = previousLayer
        self.hiddenAdvModelOutputs[9] = previousAdvLayer


        #pooling
        self.poolingLayers[3] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[3](previousLayer)
        previousAdvLayer = self.poolingLayers[3](previousAdvLayer)

        #layer 10
        self.hiddenLayers[10] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[10] = self.hiddenLayers[10]
        previousLayer = self.hiddenLayers[10](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[10](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[10] = previousLayer
        self.hiddenAdvModelOutputs[10] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 11
        self.hiddenLayers[11] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[11] = self.hiddenLayers[11]
        previousLayer = self.hiddenLayers[11](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[11](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[11] = previousLayer
        self.hiddenAdvModelOutputs[11] = previousAdvLayer

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #layer 12
        self.hiddenLayers[12] = layers.Conv2D(512, (3,3), kernel_regularizer=regularizers.l2(5e-5), activation=self.chosenActivation, padding='same')
        self.hiddenAdvLayers[12] = self.hiddenLayers[12]
        previousLayer = self.hiddenLayers[12](previousLayer)
        previousAdvLayer = self.hiddenAdvLayers[12](previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[12] = previousLayer
        self.hiddenAdvModelOutputs[12] = previousAdvLayer

        # pooling
        self.poolingLayers[4] = layers.MaxPool2D((2, 2), padding='same')
        previousLayer = self.poolingLayers[4](previousLayer)
        previousAdvLayer = self.poolingLayers[4](previousAdvLayer)

        #dense layers
        previousLayer = layers.Flatten()(previousLayer)
        previousAdvLayer = layers.Flatten()(previousAdvLayer)

        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        self.penultimateDenseLayer = layers.Dense(512, activation=self.chosenActivation, kernel_regularizer=regularizers.l2(5e-5))
        self.hiddenLayers[13] = self.penultimateDenseLayer
        self.hiddenAdvLayers[13] = self.penultimateDenseLayer
        previousLayer = self.penultimateDenseLayer(previousLayer)
        previousAdvLayer = self.penultimateDenseLayer(previousAdvLayer)
        previousLayer = layers.BatchNormalization()(previousLayer)
        previousAdvLayer = layers.BatchNormalization()(previousAdvLayer)
        self.hiddenModelOutputs[13] = previousLayer
        self.hiddenAdvModelOutputs[13] = previousAdvLayer
        if (self.dropoutRate > 0):
            previousLayer = Dropout(dropout_rate)(previousLayer)
            previousAdvLayer = Dropout(dropout_rate)(previousAdvLayer)

        #add the output layer
        #size constrained by dimensionality of inputs
        # self.logitsLayer = layers.Dense(output_dimension, activation=None, name='logitsLayer')
        # self.penultimateLayer = self.logitsActivity = self.logitsLayer(previousLayer)
        # self.penultimateAdvLayer = advLogitsActivity = self.logitsLayer(previousAdvLayer)
        self.outputLayer = layers.Dense(output_dimension, activation='softmax', name='outputLayer')#layers.Softmax()
        self.outputActivity = self.outputLayer(previousLayer)
        self.advOutputActivity = self.outputLayer(previousAdvLayer)
        # self.hiddenModelOutputs[14] = self.outputActivity
        # self.hiddenAdvLayers[14] = self.advOutputActivity
        #set up the logits layer (not just breaking apart the outputlayer because we want to be able to read in old pretrained models, so we'll just invert for this
        #softmax^-1 (X) at coordinate i = log(X_i) - log(\sum_j exp(X_j))
        self.penultimateLayer = self.logitsActivity = K.log(self.outputActivity) + K.log(K.sum(K.exp(self.outputActivity)))
        self.advPenultimateLayer = self.advLogitsActivity = K.log(self.advOutputActivity) + K.log(K.sum(K.exp(self.advOutputActivity)))
        # self.hiddenModelOutputs[14] = self.penultimateLayer
        # self.hiddenAdvLayers[14] = self.advPenultimateLayer
        # # setup the models with which we can see states of hidden layers
        numberOfHiddenLayers = len(self.hiddenLayers)-1
        # self.hiddenModelOutputs, self.hiddenAdvModelOutputs = dict(), dict()
        # for curLayer in range(numberOfHiddenLayers):
        #     self.hiddenModelOutputs[curLayer] = self.hiddenLayers[curLayer].output
        #     self.hiddenAdvModelOutputs[curLayer] = self.hiddenAdvLayers[curLayer].output

        #collect adversarial projections and benign projections

        benProjs = layers.concatenate([layers.Flatten()(self.hiddenModelOutputs[curLayer]) for curLayer in range(numberOfHiddenLayers)])
        advProjs = layers.concatenate([layers.Flatten()(self.hiddenAdvModelOutputs[curLayer]) for curLayer in range(numberOfHiddenLayers)])
        self.benProjs, self.advProjs = benProjs, advProjs

        #define our custom loss function depending on how the intializer wants to regularize (i.e., the "reg" argument)
        #this is cross_entropy + \sum_layers(abs(benign_projection-adv_projection))
        self.unprotected = unprotected
        self.reg = reg
        if (unprotected):
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
            elif (reg ==  'SYMKL'):
                def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        alpha=1.01
                        X = Xdata = benProjs
                        Y = Ydata = advProjs

                        #center the responses
                        #X = (X - tf.math.reduce_mean(X))/tf.math.reduce_std(X)
                        #Y = (Y - tf.math.reduce_mean(Y))/tf.math.reduce_std(Y)

                        #initialize some parameters
                        numberOfSamplesX = tf.shape(Xdata)[0]
                        # if numberOfSamplesX is None:
                        #     numberOfSamplesX = 1
                        # if (len(Xdata.shape) == 1):
                        #     numberOfFeaturesX = 1
                        # else:
                        numberOfFeaturesX = tf.shape(Xdata)[1]

                        numberOfSamplesY = tf.shape(Ydata)[0]
                        # if numberOfSamplesY is None:
                        #     numberOfSamplesY = 1
                        # if (len(Ydata.shape) == 1):
                        #     numberOfFeaturesY = 1
                        # else:
                        numberOfFeaturesY = tf.shape(Ydata)[1]
                        ################################################################################################################
                        # calculate sigmas
                        sigmaX = 5 * tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))
                        sigmaY = 5 * tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))

                        # calculate H_x, entropy of X
                        G = tf.cast(tf.linalg.matmul(X, tf.transpose(X)), dtype=tf.float32)
                        diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
                        # # -(1/(2*np.power(sigma,2)))*(diagG*np.ones((1,numberOfSamples))+np.ones((numberOfSamples,1))*diagG.T - 2*G)
                        # intoExp = (-1/(2*tf.pow(sigmaX, 2)))*(tf.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.cast(tf.ones((1, numberOfSamplesX)), tf.float32)) + tf.matmul(tf.cast(tf.ones((numberOfSamplesX, 1)), tf.float32), tf.reshape(diagG, (1, numberOfSamplesX))) - 2*G)
                        intoExp = tf.cast((-(1/(2*tf.pow(sigmaX,2)))),dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.ones((1,numberOfSamplesX), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesX,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesX))) - 2*G)
                        Gk = tf.math.exp(intoExp)
                        K_x = tf.math.real(Gk/tf.cast(numberOfSamplesX, dtype=tf.float32))
                        #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
                        K_x = K_x + 1e-5*tf.eye(numberOfSamplesX)#0.5*(K_x + tf.transpose(K_x))
                        # K_x = tf.where(K_x > 1e-10, K_x, K_x+1e-10*tf.zeros_like(K_x))
                        L_x,L_xv = tf.linalg.eigh(K_x)

                        lambda_x = tf.abs(L_x)
                        expLambdaX = lambda_x
                        #
                        # calculate H_y, entropy of Y
                        G = tf.cast(tf.matmul(Y, tf.transpose(Y)), dtype=tf.float32)
                        diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
                        # intoExp = (-1 / (2 * tf.pow(sigmaY, 2))) * (
                        #         diagG * tf.ones((1, numberOfSamplesY)) + tf.ones((numberOfSamplesY, 1)) * tf.transpose(diagG) - 2 * G)
                        intoExp = tf.cast((-1/(2*tf.pow(sigmaY,2))), dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesY, 1)),tf.ones((1,numberOfSamplesY), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesY,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesY))) - 2*G)
                        Gk = tf.exp(intoExp)
                        K_y = tf.math.real(Gk/tf.cast(numberOfSamplesY, dtype=tf.float32))
                        #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
                        #K_y = tf.where(K_y > 1e-10, K_y, K_y+1e-10*tf.zeros_like(K_y))
                        K_y = K_y + 1e-5*tf.eye(numberOfSamplesY)

                        L_y, L_yv= tf.linalg.eigh(K_y)
                        lambda_y = K.abs(L_y)
                        # nonZeroLYs = K.where(lambda_y > 0.01)
                        expLambdaY = lambda_y
                        # H_x = tf.cond(H_x < 0, lambda: tf.constant(0, dtype=tf.float32), lambda: H_x)
                        expLambdaY = tf.where(lambda_y > 1e-5, K.pow(lambda_y, 1-alpha), tf.zeros_like(lambda_y))
                        expLambdaX = tf.where(lambda_x > 1e-5, K.pow(lambda_x, 1-alpha), tf.zeros_like(lambda_x))
                        # expLambdaY[nonZeroLYs] = K.pow(lambda_y[nonZeroLYs], (1-alpha)*tf.zeros_like(lambda_y[nonZeroLYs]))
                        # expLambdaY = tf.cond(lambda_y > 0, lambda x: K.pow(x, 1-alpha), lambda x: 0)#K.pow(lambda_y[nonZeroLYs], 1 - alpha)
                        crossEigs0 = tf.multiply(K.pow(lambda_x, alpha), expLambdaY)
                        crossEigs1 = tf.multiply(K.pow(lambda_y, alpha), expLambdaX)
                        Rdiv0 = (1 / (alpha - 1)) * K.log(K.sum(K.abs(crossEigs0)))
                        Rdiv1 = (1 / (alpha - 1)) * K.log(K.sum(K.abs(crossEigs1)))
                        Rdiv = 0.5*(Rdiv0+Rdiv1)
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))
                        return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*Rdiv
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))/(tf.cast(K.sum(K.square(benProjs)), tf.float32))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))/(tf.cast(K.sum(K.square(benProjs)), tf.float32))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(K.sum(K.abs(benProjs)+0.0000000001), tf.float32))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(tf.shape(benProjs)[0], tf.float32))
                    return customLoss
            # online and busted
            # elif (reg ==  'KL'):
            #     def customLossWrapper(benProjs, advProjs, penaltyCoeffKL = self.penaltyCoeff):
            #         def customLoss(y_true, y_pred):
            #             alpha=1.01
            #             X = Xdata = benProjs
            #             Y = Ydata = advProjs

            #             #center the responses
            #             #X = (X - tf.math.reduce_mean(X))/tf.math.reduce_std(X)
            #             #Y = (Y - tf.math.reduce_mean(Y))/tf.math.reduce_std(Y)

            #             #initialize some parameters
            #             numberOfSamplesX = tf.shape(Xdata)[0]
            #             # if numberOfSamplesX is None:
            #             #     numberOfSamplesX = 1
            #             # if (len(Xdata.shape) == 1):
            #             #     numberOfFeaturesX = 1
            #             # else:
            #             numberOfFeaturesX = tf.shape(Xdata)[1]

            #             numberOfSamplesY = tf.shape(Ydata)[0]
            #             # if numberOfSamplesY is None:
            #             #     numberOfSamplesY = 1
            #             # if (len(Ydata.shape) == 1):
            #             #     numberOfFeaturesY = 1
            #             # else:
            #             numberOfFeaturesY = tf.shape(Ydata)[1]
            #             ################################################################################################################
            #             # calculate sigmas
            #             sigmaX = 5 * tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))
            #             sigmaY = 5 * tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))

            #             # calculate H_x, entropy of X
            #             G = tf.cast(tf.linalg.matmul(X, tf.transpose(X)), dtype=tf.float32)
            #             diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
            #             # # -(1/(2*np.power(sigma,2)))*(diagG*np.ones((1,numberOfSamples))+np.ones((numberOfSamples,1))*diagG.T - 2*G)
            #             # intoExp = (-1/(2*tf.pow(sigmaX, 2)))*(tf.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.cast(tf.ones((1, numberOfSamplesX)), tf.float32)) + tf.matmul(tf.cast(tf.ones((numberOfSamplesX, 1)), tf.float32), tf.reshape(diagG, (1, numberOfSamplesX))) - 2*G)
            #             intoExp = tf.cast((-(1/(2*tf.pow(sigmaX,2)))),dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.ones((1,numberOfSamplesX), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesX,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesX))) - 2*G)
            #             Gk = tf.math.exp(intoExp)
            #             K_x = tf.math.real(Gk/tf.cast(numberOfSamplesX, dtype=tf.float32))
            #             #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
            #             #K_x = K_x + 1e-5#0.5*(K_x + tf.transpose(K_x))
            #             K_x = K_x + 1e-5*tf.eye(numberOfSamplesX)

            #             # K_x = tf.where(K_x > 1e-10, K_x, K_x+1e-10*tf.zeros_like(K_x))
            #             L_x,L_xv = tf.linalg.eigh(K_x)

            #             lambda_x = tf.abs(L_x)
            #             #
            #             # calculate H_y, entropy of Y
            #             G = tf.cast(tf.matmul(Y, tf.transpose(Y)), dtype=tf.float32)
            #             diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
            #             # intoExp = (-1 / (2 * tf.pow(sigmaY, 2))) * (
            #             #         diagG * tf.ones((1, numberOfSamplesY)) + tf.ones((numberOfSamplesY, 1)) * tf.transpose(diagG) - 2 * G)
            #             intoExp = tf.cast((-1/(2*tf.pow(sigmaY,2))), dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesY, 1)),tf.ones((1,numberOfSamplesY), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesY,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesY))) - 2*G)
            #             Gk = tf.exp(intoExp)
            #             K_y = tf.math.real(Gk/tf.cast(numberOfSamplesY, dtype=tf.float32))
            #             #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
            #             # K_y = tf.where(K_y > 1e-10, K_y, K_y+1e-10*tf.zeros_like(K_y))
            #             K_y = K_y + 1e-5*tf.eye(numberOfSamplesY)#0.5*(K_y + tf.transpose(K_y))

            #             L_y, L_yv= tf.linalg.eigh(K_y)
            #             lambda_y = K.abs(L_y)
            #             # nonZeroLYs = K.where(lambda_y > 0.01)
            #             expLambdaY = lambda_y
            #             # H_x = tf.cond(H_x < 0, lambda: tf.constant(0, dtype=tf.float32), lambda: H_x)
            #             expLambdaY = tf.where(K.abs(lambda_y) > 1e-5, K.pow(lambda_y, 1-alpha), tf.zeros_like(lambda_y))
            #             # expLambdaY[nonZeroLYs] = K.pow(lambda_y[nonZeroLYs], (1-alpha)*tf.zeros_like(lambda_y[nonZeroLYs]))
            #             # expLambdaY = tf.cond(lambda_y > 0, lambda x: K.pow(x, 1-alpha), lambda x: 0)#K.pow(lambda_y[nonZeroLYs], 1 - alpha)
            #             crossEigs = tf.multiply(K.pow(lambda_x, alpha), expLambdaY)
            #             Rdiv = (1 / (alpha - 1)) * K.log(K.sum(K.abs(crossEigs)))
            #             # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))
            #             return (1-(penaltyCoeffKL))*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeffKL*Rdiv
            #             # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))/(tf.cast(K.sum(K.square(benProjs)), tf.float32))
            #             # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))/(tf.cast(K.sum(K.square(benProjs)), tf.float32))
            #             # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(K.sum(K.abs(benProjs)+0.0000000001), tf.float32))
            #             # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(tf.shape(benProjs)[0], tf.float32))
            #         return customLoss
            # elif (reg ==  'KLL1'):
            #     def customLossWrapper(benProjs, advProjs, penaltyCoeffKL = self.penaltyCoeff, penaltyCoeffL1 = self.penaltyCoeffL1):
            #         def customLoss(y_true, y_pred): 
            #             alpha=1.01
            #             X = Xdata = benProjs
            #             Y = Ydata = advProjs

            #             #center the responses
            #             #X = (X - tf.math.reduce_mean(X))/tf.math.reduce_std(X)
            #             #Y = (Y - tf.math.reduce_mean(Y))/tf.math.reduce_std(Y)

            #             #initialize some parameters
            #             numberOfSamplesX = tf.shape(Xdata)[0]
            #             # if numberOfSamplesX is None:
            #             #     numberOfSamplesX = 1
            #             # if (len(Xdata.shape) == 1):
            #             #     numberOfFeaturesX = 1
            #             # else:
            #             numberOfFeaturesX = tf.shape(Xdata)[1]

            #             numberOfSamplesY = tf.shape(Ydata)[0]
            #             # if numberOfSamplesY is None:
            #             #     numberOfSamplesY = 1
            #             # if (len(Ydata.shape) == 1):
            #             #     numberOfFeaturesY = 1
            #             # else:
            #             numberOfFeaturesY = tf.shape(Ydata)[1]
            #             ################################################################################################################
            #             # calculate sigmas
            #             sigmaX = 5 * tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))
            #             sigmaY = 5 * tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))

            #             # calculate H_x, entropy of X
            #             G = tf.cast(tf.linalg.matmul(X, tf.transpose(X)), dtype=tf.float32)
            #             diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
            #             # # -(1/(2*np.power(sigma,2)))*(diagG*np.ones((1,numberOfSamples))+np.ones((numberOfSamples,1))*diagG.T - 2*G)
            #             # intoExp = (-1/(2*tf.pow(sigmaX, 2)))*(tf.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.cast(tf.ones((1, numberOfSamplesX)), tf.float32)) + tf.matmul(tf.cast(tf.ones((numberOfSamplesX, 1)), tf.float32), tf.reshape(diagG, (1, numberOfSamplesX))) - 2*G)
            #             intoExp = tf.cast((-(1/(2*tf.pow(sigmaX,2)))),dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.ones((1,numberOfSamplesX), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesX,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesX))) - 2*G)
            #             Gk = tf.math.exp(intoExp)
            #             K_x = tf.math.real(Gk/tf.cast(numberOfSamplesX, dtype=tf.float32))
            #             #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
            #             #K_x = K_x + 1e-5#0.5*(K_x + tf.transpose(K_x))
            #             K_x = K_x + 1e-5*tf.eye(numberOfSamplesX)

            #             # K_x = tf.where(K_x > 1e-10, K_x, K_x+1e-10*tf.zeros_like(K_x))
            #             L_x,L_xv = tf.linalg.eigh(K_x)

            #             lambda_x = tf.abs(L_x)
            #             #
            #             # calculate H_y, entropy of Y
            #             G = tf.cast(tf.matmul(Y, tf.transpose(Y)), dtype=tf.float32)
            #             diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
            #             # intoExp = (-1 / (2 * tf.pow(sigmaY, 2))) * (
            #             #         diagG * tf.ones((1, numberOfSamplesY)) + tf.ones((numberOfSamplesY, 1)) * tf.transpose(diagG) - 2 * G)
            #             intoExp = tf.cast((-1/(2*tf.pow(sigmaY,2))), dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesY, 1)),tf.ones((1,numberOfSamplesY), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesY,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesY))) - 2*G)
            #             Gk = tf.exp(intoExp)
            #             K_y = tf.math.real(Gk/tf.cast(numberOfSamplesY, dtype=tf.float32))
            #             #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
            #             # K_y = tf.where(K_y > 1e-10, K_y, K_y+1e-10*tf.zeros_like(K_y))
            #             K_y = K_y + 1e-5*tf.eye(numberOfSamplesY)#0.5*(K_y + tf.transpose(K_y))

            #             L_y, L_yv= tf.linalg.eigh(K_y)
            #             lambda_y = K.abs(L_y)
            #             # nonZeroLYs = K.where(lambda_y > 0.01)
            #             expLambdaY = lambda_y
            #             # H_x = tf.cond(H_x < 0, lambda: tf.constant(0, dtype=tf.float32), lambda: H_x)
            #             expLambdaY = tf.where(K.abs(lambda_y) > 1e-5, K.pow(lambda_y, 1-alpha), tf.zeros_like(lambda_y))
            #             # expLambdaY[nonZeroLYs] = K.pow(lambda_y[nonZeroLYs], (1-alpha)*tf.zeros_like(lambda_y[nonZeroLYs]))
            #             # expLambdaY = tf.cond(lambda_y > 0, lambda x: K.pow(x, 1-alpha), lambda x: 0)#K.pow(lambda_y[nonZeroLYs], 1 - alpha)
            #             crossEigs = tf.multiply(K.pow(lambda_x, alpha), expLambdaY)
            #             Rdiv = (1 / (alpha - 1)) * K.log(K.sum(K.abs(crossEigs)))
            #             # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))
            #             return (1-(penaltyCoeffKL+penaltyCoeffL1))*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeffKL*Rdiv + penaltyCoeffL1*K.sum(K.abs(benProjs - advProjs))/(tf.cast(K.sum(K.abs(benProjs)), tf.float32))
            #         return customLoss
            # elif (reg ==  'KLTimesL1'):
            #     def customLossWrapper(benProjs, advProjs, penaltyCoeffKL = self.penaltyCoeff, penaltyCoeffL1 = self.penaltyCoeffL1):
            #         def customLoss(y_true, y_pred): 
            #             alpha=tf.constant(1.01, dtype=tf.float32)
            #             X = Xdata = benProjs
            #             Y = Ydata = advProjs

            #             #center the responses
            #             #X = (X - tf.math.reduce_mean(X))/tf.math.reduce_std(X)
            #             #Y = (Y - tf.math.reduce_mean(Y))/tf.math.reduce_std(Y)

            #             #initialize some parameters
            #             numberOfSamplesX = tf.cast(tf.shape(Xdata)[0],dtype=tf.int32)
            #             # if numberOfSamplesX is None:
            #             #     numberOfSamplesX = 1
            #             # if (len(Xdata.shape) == 1):
            #             #     numberOfFeaturesX = 1
            #             # else:
            #             numberOfFeaturesX = tf.cast(tf.shape(Xdata)[1], dtype=tf.int32)

            #             numberOfSamplesY = tf.cast(tf.shape(Ydata)[0], dtype=tf.int32)
            #             # if numberOfSamplesY is None:
            #             #     numberOfSamplesY = 1
            #             # if (len(Ydata.shape) == 1):
            #             #     numberOfFeaturesY = 1
            #             # else:
            #             numberOfFeaturesY = tf.cast(tf.shape(Ydata)[1], dtype=tf.int32)
            #             ################################################################################################################
            #             # calculate sigmas
            #             sigmaX = 5 * tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))
            #             sigmaY = 5 * tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))

            #             # calculate H_x, entropy of X
            #             G = tf.cast(tf.linalg.matmul(X, tf.transpose(X)), dtype=tf.float32)
            #             diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
            #             # # -(1/(2*np.power(sigma,2)))*(diagG*np.ones((1,numberOfSamples))+np.ones((numberOfSamples,1))*diagG.T - 2*G)
            #             intoExp = (-1/(2*tf.pow(sigmaX, 2)))*(tf.matmul(tf.reshape(diagG, [numberOfSamplesX, 1]),tf.cast(tf.ones([1, numberOfSamplesX]), tf.float32)) + tf.matmul(tf.cast(tf.ones([numberOfSamplesX, 1]), tf.float32), tf.reshape(diagG, [1, numberOfSamplesX])) - 2*G)
            #             # intoExp = tf.cast((-(1/(2*tf.pow(sigmaX,2)))),dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.ones((1,numberOfSamplesX), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesX,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesX))) - 2*G)
            #             Gk = tf.math.exp(intoExp)
            #             K_x = tf.math.real(Gk/tf.cast(numberOfSamplesX, dtype=tf.float32))
            #             #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
            #             #K_x = K_x + 1e-5#0.5*(K_x + tf.transpose(K_x))
            #             K_x = K_x + 1e-5*tf.eye(numberOfSamplesX)

            #             # K_x = tf.where(K_x > 1e-10, K_x, K_x+1e-10*tf.zeros_like(K_x))
            #             L_x,L_xv = tf.linalg.eigh(K_x)

            #             lambda_x = tf.abs(L_x)
            #             #
            #             # calculate H_y, entropy of Y
            #             G = tf.cast(tf.matmul(Y, tf.transpose(Y)), dtype=tf.float32)
            #             diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
            #             # intoExp = (-1 / (2 * tf.pow(sigmaY, 2))) * (
            #             #         diagG * tf.ones((1, numberOfSamplesY)) + tf.ones((numberOfSamplesY, 1)) * tf.transpose(diagG) - 2 * G)
            #             intoExp = tf.cast((-1/(2*tf.pow(sigmaY,2))), dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesY, 1)),tf.ones((1,numberOfSamplesY), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesY,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesY))) - tf.constant(2,dtype=tf.float32)*G)
            #             Gk = tf.exp(intoExp)
            #             K_y = tf.math.real(Gk/tf.cast(numberOfSamplesY, dtype=tf.float32))
            #             #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
            #             # K_y = tf.where(K_y > 1e-10, K_y, K_y+1e-10*tf.zeros_like(K_y))
            #             K_y = K_y + 1e-5*tf.eye(numberOfSamplesY)#0.5*(K_y + tf.transpose(K_y))

            #             L_y, L_yv= tf.linalg.eigh(K_y)
            #             lambda_y = K.abs(L_y)
            #             # nonZeroLYs = K.where(lambda_y > 0.01)
            #             expLambdaY = lambda_y
            #             # H_x = tf.cond(H_x < 0, lambda: tf.constant(0, dtype=tf.float32), lambda: H_x)
            #             expLambdaY = tf.where(K.abs(lambda_y) > 1e-5, K.pow(lambda_y, 1-alpha), tf.zeros_like(lambda_y))
            #             # expLambdaY[nonZeroLYs] = K.pow(lambda_y[nonZeroLYs], (1-alpha)*tf.zeros_like(lambda_y[nonZeroLYs]))
            #             # expLambdaY = tf.cond(lambda_y > 0, lambda x: K.pow(x, 1-alpha), lambda x: 0)#K.pow(lambda_y[nonZeroLYs], 1 - alpha)
            #             crossEigs = tf.multiply(K.pow(lambda_x, alpha), expLambdaY)
            #             Rdiv = (1 / (alpha - 1)) * K.log(K.sum(K.abs(crossEigs)))
            #             # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))
            #             cosine_loss = tf.keras.losses.CosineSimilarity(axis=0)
            #             return (1-(penaltyCoeffKL))*K.categorical_crossentropy(y_true, y_pred) + (cosine_loss(benProjs, advProjs))*(penaltyCoeffKL*Rdiv)#*(penaltyCoeffL1*K.sum(K.abs(benProjs - advProjs))/(tf.cast(K.sum(K.abs(benProjs)), tf.float32)))
            #         return customLoss
            

            elif (reg ==  'infosep'):
                def customLossWrapper(benProjs, advProjs, penaltyCoeffKL = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        alpha=tf.constant(1.01)
                        X = Xdata = benProjs
                        Y = Ydata = advProjs

                        #center the responses
                        #X = (X - tf.math.reduce_mean(X))/tf.math.reduce_std(X)
                        #Y = (Y - tf.math.reduce_mean(Y))/tf.math.reduce_std(Y)

                        #initialize some parameters
                        numberOfSamplesX = tf.shape(Xdata)[0]
                        # if numberOfSamplesX is None:
                        #     numberOfSamplesX = 1
                        # if (len(Xdata.shape) == 1):
                        #     numberOfFeaturesX = 1
                        # else:
                        numberOfFeaturesX = tf.shape(Xdata)[1]

                        numberOfSamplesY = tf.shape(Ydata)[0]
                        # if numberOfSamplesY is None:
                        #     numberOfSamplesY = 1
                        # if (len(Ydata.shape) == 1):
                        #     numberOfFeaturesY = 1
                        # else:
                        numberOfFeaturesY = tf.shape(Ydata)[1]
                        ################################################################################################################
                        # calculate sigmas
                        # sigmaX = tf.pow((4/(tf.cast(numberOfSamplesX, tf.float32))), tf.cast(1/(4+numberOfFeaturesX), tf.float32))*tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))*tf.math.reduce_std(X) #was hardcoded *5
                        # sigmaY = tf.pow((4/(tf.cast(numberOfSamplesY, tf.float32))), tf.cast(1/(4+numberOfFeaturesY), tf.float32))*tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))*tf.math.reduce_std(Y)
                        sigmaX = 5 * tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))
                        sigmaY = 5 * tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))

                        # calculate H_x, entropy of X
                        G = tf.cast(tf.linalg.matmul(X, tf.transpose(X)), dtype=tf.float32)
                        diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
                        # # -(1/(2*np.power(sigma,2)))*(diagG*np.ones((1,numberOfSamples))+np.ones((numberOfSamples,1))*diagG.T - 2*G)
                        # intoExp = (-1/(2*tf.pow(sigmaX, 2)))*(tf.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.cast(tf.ones((1, numberOfSamplesX)), tf.float32)) + tf.matmul(tf.cast(tf.ones((numberOfSamplesX, 1)), tf.float32), tf.reshape(diagG, (1, numberOfSamplesX))) - 2*G)
                        intoExp = tf.multiply(tf.cast((-(1/(2*tf.pow(sigmaX,2)))),dtype=tf.float32),(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.ones((1,numberOfSamplesX), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesX,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesX)))) - 2*G)
                        Gk = tf.math.exp(intoExp)
                        K_x = tf.math.real(Gk/tf.cast(numberOfSamplesX, dtype=tf.float32))
                        #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
                        #K_x = K_x + 1e-5#0.5*(K_x + tf.transpose(K_x))
                        randomHalf = tf.random.normal([numberOfSamplesX,numberOfSamplesX])
                        K_x = K_x + 1e-6*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-4*tf.eye(numberOfSamplesX)#1e-5*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-3*tf.eye(numberOfSamplesX)

                        # K_x = tf.where(K_x > 1e-10, K_x, K_x+1e-10*tf.zeros_like(K_x))
                        L_x,L_xv = tf.linalg.eigh(K_x)

                        lambda_x = tf.abs(L_x)
                        #
                        # calculate H_y, entropy of Y
                        G = tf.cast(tf.matmul(Y, tf.transpose(Y)), dtype=tf.float32)
                        diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
                        # intoExp = (-1 / (2 * tf.pow(sigmaY, 2))) * (
                        #         diagG * tf.ones((1, numberOfSamplesY)) + tf.ones((numberOfSamplesY, 1)) * tf.transpose(diagG) - 2 * G)
                        intoExp = tf.cast((-1/(2*tf.pow(sigmaY,2))), dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesY, 1)),tf.ones((1,numberOfSamplesY), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesY,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesY))) - 2*G)
                        Gk = tf.exp(intoExp)
                        K_y = tf.math.real(Gk/tf.cast(numberOfSamplesY, dtype=tf.float32))
                        #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
                        # K_y = tf.where(K_y > 1e-10, K_y, K_y+1e-10*tf.zeros_like(K_y))
                        randomHalf = tf.random.normal([numberOfSamplesY, numberOfSamplesY])
                        K_y = K_y + 1e-6*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-4*tf.eye(numberOfSamplesY)#0.5*(K_y + tf.transpose(K_y))#tf.math.abs((randomHalf+tf.transpose(randomHalf)))

                        L_y, L_yv= tf.linalg.eigh(K_y)
                        lambda_y = K.abs(L_y)
                        # nonZeroLYs = K.where(lambda_y > 0.01)
                        # expLambdaY = lambda_y
                        # H_x = tf.cond(H_x < 0, lambda: tf.constant(0, dtype=tf.float32), lambda: H_x)
                        expLambdaY = tf.where(K.abs(lambda_y) > 1e-6, K.pow(lambda_y, 1-alpha), tf.ones_like(lambda_y))
                        # expLambdaY[nonZeroLYs] = K.pow(lambda_y[nonZeroLYs], (1-alpha)*tf.zeros_like(lambda_y[nonZeroLYs]))
                        # expLambdaY = tf.cond(lambda_y > 0, lambda x: K.pow(x, 1-alpha), lambda x: 0)#K.pow(lambda_y[nonZeroLYs], 1 - alpha)
                        crossEigs = tf.multiply(K.pow(lambda_x, alpha), expLambdaY)
                        Rdiv = (1 / (alpha - 1)) * K.log(K.sum(K.abs(crossEigs)))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))
                        return (1-(penaltyCoeffKL))*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeffKL*Rdiv
                        #todo put info sep objective here
                    return customLoss

            elif (reg ==  'KL'):
                def customLossWrapper(benProjs, advProjs, penaltyCoeffKL = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        alpha=tf.constant(1.01)
                        X = Xdata = benProjs
                        Y = Ydata = advProjs

                        #center the responses
                        #X = (X - tf.math.reduce_mean(X))/tf.math.reduce_std(X)
                        #Y = (Y - tf.math.reduce_mean(Y))/tf.math.reduce_std(Y)

                        #initialize some parameters
                        numberOfSamplesX = tf.shape(Xdata)[0]
                        # if numberOfSamplesX is None:
                        #     numberOfSamplesX = 1
                        # if (len(Xdata.shape) == 1):
                        #     numberOfFeaturesX = 1
                        # else:
                        numberOfFeaturesX = tf.shape(Xdata)[1]

                        numberOfSamplesY = tf.shape(Ydata)[0]
                        # if numberOfSamplesY is None:
                        #     numberOfSamplesY = 1
                        # if (len(Ydata.shape) == 1):
                        #     numberOfFeaturesY = 1
                        # else:
                        numberOfFeaturesY = tf.shape(Ydata)[1]
                        ################################################################################################################
                        # calculate sigmas
                        # sigmaX = tf.pow((4/(tf.cast(numberOfSamplesX, tf.float32))), tf.cast(1/(4+numberOfFeaturesX), tf.float32))*tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))*tf.math.reduce_std(X) #was hardcoded *5
                        # sigmaY = tf.pow((4/(tf.cast(numberOfSamplesY, tf.float32))), tf.cast(1/(4+numberOfFeaturesY), tf.float32))*tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))*tf.math.reduce_std(Y)
                        sigmaX = 5 * tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))
                        sigmaY = 5 * tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))

                        # calculate H_x, entropy of X
                        G = tf.cast(tf.linalg.matmul(X, tf.transpose(X)), dtype=tf.float32)
                        diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
                        # # -(1/(2*np.power(sigma,2)))*(diagG*np.ones((1,numberOfSamples))+np.ones((numberOfSamples,1))*diagG.T - 2*G)
                        # intoExp = (-1/(2*tf.pow(sigmaX, 2)))*(tf.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.cast(tf.ones((1, numberOfSamplesX)), tf.float32)) + tf.matmul(tf.cast(tf.ones((numberOfSamplesX, 1)), tf.float32), tf.reshape(diagG, (1, numberOfSamplesX))) - 2*G)
                        intoExp = tf.multiply(tf.cast((-(1/(2*tf.pow(sigmaX,2)))),dtype=tf.float32),(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.ones((1,numberOfSamplesX), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesX,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesX)))) - 2*G)
                        Gk = tf.math.exp(intoExp)
                        K_x = tf.math.real(Gk/tf.cast(numberOfSamplesX, dtype=tf.float32))
                        #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
                        #K_x = K_x + 1e-5#0.5*(K_x + tf.transpose(K_x))
                        randomHalf = tf.random.normal([numberOfSamplesX,numberOfSamplesX])
                        K_x = K_x + 1e-6*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-4*tf.eye(numberOfSamplesX)#1e-5*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-3*tf.eye(numberOfSamplesX)

                        # K_x = tf.where(K_x > 1e-10, K_x, K_x+1e-10*tf.zeros_like(K_x))
                        L_x,L_xv = tf.linalg.eigh(K_x)

                        lambda_x = tf.abs(L_x)
                        #
                        # calculate H_y, entropy of Y
                        G = tf.cast(tf.matmul(Y, tf.transpose(Y)), dtype=tf.float32)
                        diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
                        # intoExp = (-1 / (2 * tf.pow(sigmaY, 2))) * (
                        #         diagG * tf.ones((1, numberOfSamplesY)) + tf.ones((numberOfSamplesY, 1)) * tf.transpose(diagG) - 2 * G)
                        intoExp = tf.cast((-1/(2*tf.pow(sigmaY,2))), dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesY, 1)),tf.ones((1,numberOfSamplesY), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesY,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesY))) - 2*G)
                        Gk = tf.exp(intoExp)
                        K_y = tf.math.real(Gk/tf.cast(numberOfSamplesY, dtype=tf.float32))
                        #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
                        # K_y = tf.where(K_y > 1e-10, K_y, K_y+1e-10*tf.zeros_like(K_y))
                        randomHalf = tf.random.normal([numberOfSamplesY, numberOfSamplesY])
                        K_y = K_y + 1e-6*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-4*tf.eye(numberOfSamplesY)#0.5*(K_y + tf.transpose(K_y))#tf.math.abs((randomHalf+tf.transpose(randomHalf)))

                        L_y, L_yv= tf.linalg.eigh(K_y)
                        lambda_y = K.abs(L_y)
                        # nonZeroLYs = K.where(lambda_y > 0.01)
                        # expLambdaY = lambda_y
                        # H_x = tf.cond(H_x < 0, lambda: tf.constant(0, dtype=tf.float32), lambda: H_x)
                        expLambdaY = tf.where(K.abs(lambda_y) > 1e-6, K.pow(lambda_y, 1-alpha), tf.ones_like(lambda_y))
                        # expLambdaY[nonZeroLYs] = K.pow(lambda_y[nonZeroLYs], (1-alpha)*tf.zeros_like(lambda_y[nonZeroLYs]))
                        # expLambdaY = tf.cond(lambda_y > 0, lambda x: K.pow(x, 1-alpha), lambda x: 0)#K.pow(lambda_y[nonZeroLYs], 1 - alpha)
                        crossEigs = tf.multiply(K.pow(lambda_x, alpha), expLambdaY)
                        Rdiv = (1 / (alpha - 1)) * K.log(K.sum(K.abs(crossEigs)))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))
                        return (1-(penaltyCoeffKL))*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeffKL*Rdiv
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))/(tf.cast(K.sum(K.square(benProjs)), tf.float32))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))/(tf.cast(K.sum(K.square(benProjs)), tf.float32))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(K.sum(K.abs(benProjs)+0.0000000001), tf.float32))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(tf.shape(benProjs)[0], tf.float32))
                    return customLoss
            elif (reg ==  'KLL1'):
                def customLossWrapper(benProjs, advProjs, penaltyCoeffKL = self.penaltyCoeff, penaltyCoeffL1 = self.penaltyCoeffL1):
                    def customLoss(y_true, y_pred): 
                        alpha=tf.constant(1.01)
                        X = Xdata = benProjs
                        Y = Ydata = advProjs

                        #center the responses
                        #X = (X - tf.math.reduce_mean(X))/tf.math.reduce_std(X)
                        #Y = (Y - tf.math.reduce_mean(Y))/tf.math.reduce_std(Y)

                        #initialize some parameters
                        numberOfSamplesX = tf.shape(Xdata)[0]
                        # if numberOfSamplesX is None:
                        #     numberOfSamplesX = 1
                        # if (len(Xdata.shape) == 1):
                        #     numberOfFeaturesX = 1
                        # else:
                        numberOfFeaturesX = tf.shape(Xdata)[1]

                        numberOfSamplesY = tf.shape(Ydata)[0]
                        # if numberOfSamplesY is None:
                        #     numberOfSamplesY = 1
                        # if (len(Ydata.shape) == 1):
                        #     numberOfFeaturesY = 1
                        # else:
                        numberOfFeaturesY = tf.shape(Ydata)[1]
                        ################################################################################################################
                        # calculate sigmas
                        # sigmaX = tf.pow((4/(tf.cast(numberOfSamplesX, tf.float32))), tf.cast(1/(4+numberOfFeaturesX), tf.float32))*tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))*tf.math.reduce_std(X) #was hardcoded *5
                        # sigmaY = tf.pow((4/(tf.cast(numberOfSamplesY, tf.float32))), tf.cast(1/(4+numberOfFeaturesY), tf.float32))*tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))*tf.math.reduce_std(Y)
                        sigmaX = 5 * tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))
                        sigmaY = 5 * tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))

                        # calculate H_x, entropy of X
                        G = tf.cast(tf.linalg.matmul(X, tf.transpose(X)), dtype=tf.float32)
                        diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
                        # # -(1/(2*np.power(sigma,2)))*(diagG*np.ones((1,numberOfSamples))+np.ones((numberOfSamples,1))*diagG.T - 2*G)
                        # intoExp = (-1/(2*tf.pow(sigmaX, 2)))*(tf.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.cast(tf.ones((1, numberOfSamplesX)), tf.float32)) + tf.matmul(tf.cast(tf.ones((numberOfSamplesX, 1)), tf.float32), tf.reshape(diagG, (1, numberOfSamplesX))) - 2*G)
                        intoExp = tf.multiply(tf.cast((-(1/(2*tf.pow(sigmaX,2)))),dtype=tf.float32),(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.ones((1,numberOfSamplesX), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesX,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesX)))) - 2*G)
                        Gk = tf.math.exp(intoExp)
                        K_x = tf.math.real(Gk/tf.cast(numberOfSamplesX, dtype=tf.float32))
                        #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
                        #K_x = K_x + 1e-5#0.5*(K_x + tf.transpose(K_x))
                        randomHalf = tf.random.normal([numberOfSamplesX,numberOfSamplesX])
                        K_x = K_x + 1e-6*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-4*tf.eye(numberOfSamplesX)#1e-5*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-3*tf.eye(numberOfSamplesX)

                        # K_x = tf.where(K_x > 1e-10, K_x, K_x+1e-10*tf.zeros_like(K_x))
                        L_x,L_xv = tf.linalg.eigh(K_x)

                        lambda_x = tf.abs(L_x)
                        #
                        # calculate H_y, entropy of Y
                        G = tf.cast(tf.matmul(Y, tf.transpose(Y)), dtype=tf.float32)
                        diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
                        # intoExp = (-1 / (2 * tf.pow(sigmaY, 2))) * (
                        #         diagG * tf.ones((1, numberOfSamplesY)) + tf.ones((numberOfSamplesY, 1)) * tf.transpose(diagG) - 2 * G)
                        intoExp = tf.cast((-1/(2*tf.pow(sigmaY,2))), dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesY, 1)),tf.ones((1,numberOfSamplesY), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesY,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesY))) - 2*G)
                        Gk = tf.exp(intoExp)
                        K_y = tf.math.real(Gk/tf.cast(numberOfSamplesY, dtype=tf.float32))
                        #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
                        # K_y = tf.where(K_y > 1e-10, K_y, K_y+1e-10*tf.zeros_like(K_y))
                        randomHalf = tf.random.normal([numberOfSamplesY, numberOfSamplesY])
                        K_y = K_y + 1e-6*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-4*tf.eye(numberOfSamplesY)#0.5*(K_y + tf.transpose(K_y))#tf.math.abs((randomHalf+tf.transpose(randomHalf)))

                        L_y, L_yv= tf.linalg.eigh(K_y)
                        lambda_y = K.abs(L_y)
                        # nonZeroLYs = K.where(lambda_y > 0.01)
                        # expLambdaY = lambda_y
                        # H_x = tf.cond(H_x < 0, lambda: tf.constant(0, dtype=tf.float32), lambda: H_x)
                        expLambdaY = tf.where(K.abs(lambda_y) > 1e-6, K.pow(lambda_y, 1-alpha), tf.cast(1/(numberOfSamplesY), dtype=tf.float32)*tf.ones_like(lambda_y))
                        # expLambdaY[nonZeroLYs] = K.pow(lambda_y[nonZeroLYs], (1-alpha)*tf.zeros_like(lambda_y[nonZeroLYs]))
                        # expLambdaY = tf.cond(lambda_y > 0, lambda x: K.pow(x, 1-alpha), lambda x: 0)#K.pow(lambda_y[nonZeroLYs], 1 - alpha)
                        crossEigs = tf.multiply(K.pow(lambda_x, alpha), expLambdaY)
                        Rdiv = (1 / (alpha - 1)) * K.log(K.sum(K.abs(crossEigs)))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))
                        return (1-(penaltyCoeffKL+penaltyCoeffL1))*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeffKL*Rdiv + penaltyCoeffL1*K.sum(K.abs(benProjs - advProjs))/(tf.cast(K.sum(K.abs(benProjs)), tf.float32))
                    return customLoss
            elif (reg ==  'KLTimesL1'):
                def customLossWrapper(benProjs, advProjs, penaltyCoeffKL = self.penaltyCoeff, penaltyCoeffL1 = self.penaltyCoeffL1):
                    def customLoss(y_true, y_pred): 
                        alpha=tf.constant(1.01)
                        X = Xdata = benProjs
                        Y = Ydata = advProjs

                        #center the responses
                        #X = (X - tf.math.reduce_mean(X))/tf.math.reduce_std(X)
                        #Y = (Y - tf.math.reduce_mean(Y))/tf.math.reduce_std(Y)

                        #initialize some parameters
                        numberOfSamplesX = tf.shape(Xdata)[0]
                        # if numberOfSamplesX is None:
                        #     numberOfSamplesX = 1
                        # if (len(Xdata.shape) == 1):
                        #     numberOfFeaturesX = 1
                        # else:
                        numberOfFeaturesX = tf.shape(Xdata)[1]

                        numberOfSamplesY = tf.shape(Ydata)[0]
                        # if numberOfSamplesY is None:
                        #     numberOfSamplesY = 1
                        # if (len(Ydata.shape) == 1):
                        #     numberOfFeaturesY = 1
                        # else:
                        numberOfFeaturesY = tf.shape(Ydata)[1]
                        ################################################################################################################
                        # calculate sigmas
                        # sigmaX = tf.pow((4/(tf.cast(numberOfSamplesX, tf.float32))), tf.cast(1/(4+numberOfFeaturesX), tf.float32))*tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))*tf.math.reduce_std(X) #was hardcoded *5
                        # sigmaY = tf.pow((4/(tf.cast(numberOfSamplesY, tf.float32))), tf.cast(1/(4+numberOfFeaturesY), tf.float32))*tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))*tf.math.reduce_std(Y)
                        sigmaX = 5 * tf.pow(tf.cast(numberOfSamplesX, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesX), tf.float32))
                        sigmaY = 5 * tf.pow(tf.cast(numberOfSamplesY, tf.float32), tf.cast(-1 / (4 + numberOfFeaturesY), tf.float32))

                        # calculate H_x, entropy of X
                        G = tf.cast(tf.linalg.matmul(X, tf.transpose(X)), dtype=tf.float32)
                        diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
                        # # -(1/(2*np.power(sigma,2)))*(diagG*np.ones((1,numberOfSamples))+np.ones((numberOfSamples,1))*diagG.T - 2*G)
                        # intoExp = (-1/(2*tf.pow(sigmaX, 2)))*(tf.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.cast(tf.ones((1, numberOfSamplesX)), tf.float32)) + tf.matmul(tf.cast(tf.ones((numberOfSamplesX, 1)), tf.float32), tf.reshape(diagG, (1, numberOfSamplesX))) - 2*G)
                        intoExp = tf.multiply(tf.cast((-(1/(2*tf.pow(sigmaX,2)))),dtype=tf.float32),(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesX, 1)),tf.ones((1,numberOfSamplesX), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesX,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesX)))) - 2*G)
                        Gk = tf.math.exp(intoExp)
                        K_x = tf.math.real(Gk/tf.cast(numberOfSamplesX, dtype=tf.float32))
                        #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
                        #K_x = K_x + 1e-5#0.5*(K_x + tf.transpose(K_x))
                        randomHalf = tf.random.normal([numberOfSamplesX,numberOfSamplesX])
                        K_x = K_x + 1e-6*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-4*tf.eye(numberOfSamplesX)#1e-5*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-3*tf.eye(numberOfSamplesX)

                        # K_x = tf.where(K_x > 1e-10, K_x, K_x+1e-10*tf.zeros_like(K_x))
                        L_x,L_xv = tf.linalg.eigh(K_x)

                        lambda_x = tf.abs(L_x)
                        #
                        # calculate H_y, entropy of Y
                        G = tf.cast(tf.matmul(Y, tf.transpose(Y)), dtype=tf.float32)
                        diagG = tf.transpose(tf.cast(tf.linalg.tensor_diag_part(G), tf.float32))
                        # intoExp = (-1 / (2 * tf.pow(sigmaY, 2))) * (
                        #         diagG * tf.ones((1, numberOfSamplesY)) + tf.ones((numberOfSamplesY, 1)) * tf.transpose(diagG) - 2 * G)
                        intoExp = tf.cast((-1/(2*tf.pow(sigmaY,2))), dtype=tf.float32)*(tf.linalg.matmul(tf.reshape(diagG, (numberOfSamplesY, 1)),tf.ones((1,numberOfSamplesY), dtype=tf.float32))+tf.linalg.matmul(tf.ones((numberOfSamplesY,1), dtype=tf.float32),tf.reshape(diagG, (1, numberOfSamplesY))) - 2*G)
                        Gk = tf.exp(intoExp)
                        K_y = tf.math.real(Gk/tf.cast(numberOfSamplesY, dtype=tf.float32))
                        #recondition K_x since gradients can only be uniquely identified for unique eigenvalues
                        # K_y = tf.where(K_y > 1e-10, K_y, K_y+1e-10*tf.zeros_like(K_y))
                        randomHalf = tf.random.normal([numberOfSamplesY, numberOfSamplesY])
                        K_y = K_y + 1e-6*tf.math.abs((randomHalf+tf.transpose(randomHalf)))#1e-4*tf.eye(numberOfSamplesY)#0.5*(K_y + tf.transpose(K_y))#tf.math.abs((randomHalf+tf.transpose(randomHalf)))

                        L_y, L_yv= tf.linalg.eigh(K_y)
                        lambda_y = K.abs(L_y)
                        # nonZeroLYs = K.where(lambda_y > 0.01)
                        # expLambdaY = lambda_y
                        # H_x = tf.cond(H_x < 0, lambda: tf.constant(0, dtype=tf.float32), lambda: H_x)
                        expLambdaY = tf.where(K.abs(lambda_y) > 1e-5, K.pow(lambda_y, 1-alpha), tf.ones_like(lambda_y))
                        # expLambdaY[nonZeroLYs] = K.pow(lambda_y[nonZeroLYs], (1-alpha)*tf.zeros_like(lambda_y[nonZeroLYs]))
                        # expLambdaY = tf.cond(lambda_y > 0, lambda x: K.pow(x, 1-alpha), lambda x: 0)#K.pow(lambda_y[nonZeroLYs], 1 - alpha)
                        crossEigs = tf.multiply(K.pow(lambda_x, alpha), expLambdaY)
                        Rdiv = (1 / (alpha - 1)) * K.log(K.sum(K.abs(crossEigs)))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))
                        cosine_loss = tf.keras.losses.CosineSimilarity(axis=0)
                        return (1-(penaltyCoeffKL))*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeffKL*(Rdiv)*(tf.math.abs(cosine_loss(benProjs, advProjs))**2)*(K.sum(K.abs(benProjs - advProjs))/(tf.cast(K.sum(K.abs(benProjs)), tf.float32)))
                    return customLoss
            
            elif (reg ==  'cossim'):
                def customLossWrapper(benProjs, advProjs, penaltyCoeffKL = self.penaltyCoeff, penaltyCoeffL1 = self.penaltyCoeffL1):
                    def customLoss(y_true, y_pred): 
                        alpha=tf.constant(1.01)
                        X = Xdata = benProjs
                        Y = Ydata = advProjs

                        #center the responses
                        #X = (X - tf.math.reduce_mean(X))/tf.math.reduce_std(X)
                        #Y = (Y - tf.math.reduce_mean(Y))/tf.math.reduce_std(Y)

                        #initialize some parameters
                        numberOfSamplesX = tf.shape(Xdata)[0]
                        # if numberOfSamplesX is None:
                        #     numberOfSamplesX = 1
                        # if (len(Xdata.shape) == 1):
                        #     numberOfFeaturesX = 1
                        # else:
                        numberOfFeaturesX = tf.shape(Xdata)[1]

                        numberOfSamplesY = tf.shape(Ydata)[0]
                        # if numberOfSamplesY is None:
                        #     numberOfSamplesY = 1
                        # if (len(Ydata.shape) == 1):
                        #     numberOfFeaturesY = 1
                        # else:
                        numberOfFeaturesY = tf.shape(Ydata)[1]
                        cosine_loss = tf.keras.losses.CosineSimilarity(axis=0)
                        return (1-(penaltyCoeffKL))*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeffKL*(tf.math.abs(cosine_loss(benProjs, advProjs)))#*K.sum(K.abs(benProjs - advProjs))/(tf.cast(K.sum(K.abs(benProjs)), tf.float32)))#(cosine_loss(benProjs, advProjs))
                    return customLoss
            
            elif (reg ==  'LZ'):
                # qBenProjs = tensorflow.quantization.quantize(tf.cast(benProjs, tf.float32), TFQUANT_MIN, TFQUANT_MAX, tf.qint32)
                # qAdvProjs = tensorflow.quantization.quantize(tf.cast(advProjs, tf.float32), TFQUANT_MIN, TFQUANT_MAX, tf.qint32)
                lzBA = lempel_ziv_complexity(tf.as_string(layers.Flatten()(K.square(benProjs-advProjs)), precision=numberOfBins))
                # lzAdv = lempel_ziv_complexity(tf.as_string(layers.Flatten()(advProjs), precision=8))
                def customLossWrapper(lzBen, lzAdv=None, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        return  (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + 0.5*lzBA
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.square(benProjs - advProjs))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + 0.5*penaltyCoeff*K.sum(K.square(benProjs - advProjs))/(tf.cast(tf.size(K.square(benProjs)), tf.float32)) + 0.5*penaltyCoeff*K.sum(tf.cast(lzBA, tf.float32))/tf.cast(tf.size(lzBA), tf.float32)
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(K.sum(K.abs(benProjs)+0.0000000001), tf.float32))
                        # return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(K.abs(benProjs - advProjs))/(tf.cast(tf.shape(benProjs)[0], tf.float32))
                    return customLoss
            elif (reg ==  'FIM'):
                # dS = tf.gradients(self.outputLayer, self.inputLayer)
                # dS_2 = tf.matmul(dS, tf.reshape(dS, (dS.shape[1], dS.shape[0])))
                # eigs = tf.linalg.eigvals(dS_2)
                ps = tf.divide(tf.ones(shape=(tf.shape(self.outputActivity))), tf.where(self.outputActivity > 0, self.outputActivity, 1e16*tf.ones_like(self.outputActivity)))
                def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(ps)
                    return customLoss
            elif (reg ==  'logEigen'):
                # dS = tf.gradients(self.outputLayer, self.inputLayer)
                # dS_2 = tf.matmul(dS, tf.reshape(dS, (dS.shape[1], dS.shape[0])))
                # eigs = tf.linalg.eigvals(dS_2)
                ps = tf.divide(tf.ones(shape=(tf.shape(self.outputActivity))), tf.ones_like(self.outputActivity)-tf.math.log(tf.where(self.outputActivity > 0, self.outputActivity, 1e16*tf.ones_like(self.outputActivity))))
                def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(ps)
                    return customLoss
            elif (reg ==  'logEigenlogits'):
                # dS = tf.gradients(self.outputLayer, self.inputLayer)
                # dS_2 = tf.matmul(dS, tf.reshape(dS, (dS.shape[1], dS.shape[0])))
                # eigs = tf.linalg.eigvals(dS_2)
                ps = tf.divide(tf.ones(shape=(tf.shape(self.logitsActivity))), tf.ones_like(self.logitsActivity)+tf.math.log(tf.where(self.logitsActivity > 0, self.logitsActivity, 1e16*tf.ones_like(self.logitsActivity))))
                def customLossWrapper(benProjs, advProjs, penaltyCoeff = self.penaltyCoeff):
                    def customLoss(y_true, y_pred):
                        return (1-penaltyCoeff)*K.categorical_crossentropy(y_true, y_pred) + penaltyCoeff*K.sum(ps)
                    return customLoss
            elif (reg ==  'logitFIM'):
                # dS = tf.gradients(self.outputLayer, self.inputLayer)
                # dS_2 = tf.matmul(dS, tf.reshape(dS, (dS.shape[1], dS.shape[0])))
                # eigs = tf.linalg.eigvals(dS_2)
                ps = tf.divide(tf.ones(shape=(tf.shape(self.logitsActivity))), tf.where(self.logitsActivity > 0, self.logitsActivity, 1e16*tf.ones_like(self.logitsActivity)))
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
            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
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
        self.model = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=mainOutputList, name='hlrr_vgg_16')
        #if we want to use this as a frozen model
        if (freezeWeights):
            for curWeights in range(len(self.model.layers)):
                self.model.layers[curWeights].trainable = False
        self.model.compile(loss=customLossWrapper(benProjs, advProjs, self.penaltyCoeff), metrics=['acc'], optimizer=self.sgd)
        #compile the siso model
        self.singleInSingleOutModel.compile(loss= tf.keras.losses.CategoricalCrossentropy(), metrics=['acc'], optimizer=self.sgd)

        #setup the models with which we can see states of hidden layers
        self.hiddenModel = Model(inputs=[self.inputLayer, self.advInputLayer], outputs=outputsList, name='hidden_hlrr_vgg_16')
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
                normed=False, monitor='val_loss', patience=defaultPatience, testBetweenEpochs=None, 
                inputTestingData=None, inputTestingTargets=None, inputAdvPerturbations=None,
                inputTestingAGNPerturbations=None, model_path=None, keras_batch_size=None, 
                noisePowers=None, advPowers=None, dataAugmentation=False, adversarialOrder=0):
        #if a path isn't provided by caller, just use the current time for restoring best weights from fit
        if (model_path is None):
            model_path = os.path.join('A://tmp/modelCheckpoints/', 'hlr_vgg16_'+str(int(round(time.time()*1000))))

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
        earlyStopper = EarlyStopping(monitor=monitor, mode='min', patience=patience,
                                     verbose=1, min_delta=defaultLossThreshold)
        checkpoint = ModelCheckpoint(model_path, verbose=1, monitor=monitor, save_weights_only=True,
                                     save_best_only=True, mode='auto')
        callbackList = [earlyStopper, checkpoint]
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
                                                horizontal_flip=True,  # randomly flip images
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

    def evaluate(self, inputData, targets, batchSize=None):
        #evalData = [self.normalize(inputData[0], storeStats=False), self.normalize(inputData[1], storeStats=False)]
        evalData = self.normalize(inputData[0], storeStats=False)
        fullEval = self.singleInSingleOutModel.evaluate(evalData, targets, batch_size=batchSize)
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
                        max_relu_bound=self.max_relu_bound, adv_penalty=self.advPenalty, unprotected=self.unprotected,
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


if __name__ == "__main__":
    import copy
    #define test parameters
    # define parameters
    verbose = True
    kerasBatchSize = 64
    testFraction = 0.5
    numberOfClasses = 10
    trainingSetSize = -1
    numberOfAdvSamples = 1000
    trainingEpochs = 10
    trainingSetSize = 5000#100000
    maxCarliniIts = 10000
    powers = [0.05, 0.1, 0.25, 0.5, 1, 2]
    dataset="cifar10"
    # input image dimensions
    img_rows, img_cols = 32, 32#28, 28
    inputDimension = (img_rows, img_cols, 3)
    ########################################################################################################################

    ########################################################################################################################
    # split data between train and test sets
    if (dataset == 'fashion_mnist'):
        input_shape = (img_rows, img_cols, 1)
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train_up, x_test_up = np.zeros((x_train.shape[0], 32, 32)), np.zeros((x_test.shape[0], 32, 32))
        # upscale data to 32,32 (same size as cifar)
        for i in range(x_train.shape[0]):
            x_train_up[i,:,:] = cv2.resize(x_train[i,:,:], (32, 32), interpolation = cv2.INTER_AREA)
        for i in range(x_test.shape[0]):
            x_test_up[i,:,:] = cv2.resize(x_test[i,:,:], (32, 32), interpolation = cv2.INTER_AREA)
        x_train = x_train_up
        x_test = x_test_up

        #scale and standardize (with respect to stats calculated on training set)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255.
        x_test /= 255.
        # for fashion mnist: mean:0.28654176; std:0.35317212
        # x_train = (x_train - 0.28654176) / (0.35317212)
        # x_test = (x_test - 0.28654176) / (0.35317212)

    elif (dataset == 'cifar10'):
        input_shape = (img_rows, img_cols, 3)
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

    # # restrict number of classes
    # trainXs, trainTargets = [], []
    # testXs, testTargets = [], []

    # for t in range(numberOfClasses):
    #     curClassIndicesTraining = np.where(y_train == t)[0]
    #     curClassIndicesTesting = np.where(y_test == t)[0]
    #     if (trainingSetSize == -1):
    #         # arrange training data
    #         if (dataset == 'fashion_mnist'):
    #             curXTrain = np.expand_dims(x_train[curClassIndicesTraining, :, :], axis=3)
    #             curXTest = np.expand_dims(x_test[curClassIndicesTesting, :, :], axis=3)
    #         else:
    #             curXTrain = x_train[curClassIndicesTraining, :, :]
    #             curXTest = x_test[curClassIndicesTesting, :, :]
    #         # arrange testing data
    #         # curXTest = np.squeeze(x_test[curClassIndicesTesting, :])

    #     else:
    #         if (dataset == 'fashion_mnist'):
    #             # arrange training data
    #             curXTrain = np.expand_dims(x_train[curClassIndicesTraining[:trainingSetSize], :, :], axis=3)
    #             # arrange testing data
    #             curXTest = np.expand_dims(x_test[curClassIndicesTesting[:trainingSetSize], :, :], axis=3)
    #         else:
    #             # arrange training data
    #             curXTrain = x_train[curClassIndicesTraining[:trainingSetSize], :, :]
    #             # arrange testing data
    #             curXTest = x_test[curClassIndicesTesting[:trainingSetSize], :, :]

    #     trainXs.append(curXTrain)
    #     trainTargets.append((t * np.ones([curXTrain.shape[0], 1])))

    #     testXs.append(curXTest)
    #     testTargets.append(t * np.ones([curXTest.shape[0], 1]))

    # # stack our data
    # t = 0
    # stackedData, stackedTargets = np.array([]), np.array([])
    # for t in range(numberOfClasses):
    #     if (verbose):
    #         print('current class count')
    #         print(t + 1)
    #     stackedData = np.concatenate((stackedData, trainXs[t]), axis=0) if stackedData.size > 0 else trainXs[t]
    #     stackedData = np.concatenate((stackedData, testXs[t]), axis=0)
    #     stackedTargets = np.concatenate((stackedTargets, trainTargets[t]), axis=0) if stackedTargets.size > 0 else \
    #     trainTargets[t]
    #     stackedTargets = np.concatenate((stackedTargets, testTargets[t]), axis=0)

    #since this experiment resumes after pre-training, we only want our test data, since in part 1, we trained on our training data
    # trainX, testX, trainY, testY = model_selection.train_test_split(stackedData, stackedTargets, shuffle=True,
    #                                                                 test_size=testFraction, random_state=42)

    trainX, testX, trainY, testY = copy.copy(x_train), copy.copy(x_test), copy.copy(y_train), copy.copy(y_test)
    stackedData = trainX
    stackedTargets = trainY
    # trainX, testX, trainY, testY = trainX[:trainingSetSize, :, :], testX[:trainingSetSize, :, :], trainY[:trainingSetSize], testY[:trainingSetSize]
    print('beginning test')
    output_dimension = numberOfClasses

    #split validation data
    valFrac = 0.1
    trainX, valX, trainY, valY = model_selection.train_test_split(stackedData, stackedTargets, shuffle=True,
                                                                    test_size=valFrac, random_state=42)

    ourModel = LDRVGG(inputDimension, output_dimension, number_of_classes=numberOfClasses, adv_penalty=0.005,
                       loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0.33,
                       max_relu_bound=1.1, reg="HLDR", unprotected=False, verbose=False, optimizer=None)
    ourModel.model.summary()
    # unprotectedModel = LDRVGG(inputDimension, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0.05,
    #                    loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0.33,
    #                    max_relu_bound=1.1, verbose=False, unprotected=True)
    # ourModel.model.load_weights('unprotectedCifar10VGG.h5')
    ourModel.train([trainX[:trainingSetSize, :, :, :], trainX[:trainingSetSize, :, :, :]], 
                tf.keras.utils.to_categorical(trainY[:trainingSetSize], numberOfClasses),
                [valX, valX], tf.keras.utils.to_categorical(valY, numberOfClasses),
                training_epochs=trainingEpochs, 
                normed=False, monitor='val_loss', patience=defaultPatience, testBetweenEpochs=None, 
                inputTestingData=None, inputTestingTargets=None, inputAdvPerturbations=None,
                inputTestingAGNPerturbations=None, model_path=None, keras_batch_size=kerasBatchSize, 
                noisePowers=None, advPowers=None, dataAugmentation=True, adversarialOrder=0)

    # ourModel.model.save('hlrr_complete')
    # ourModel.hiddenModel.save('hlrr_complete_hidden')
    # ourModel.storeModelToDisk('hlrr_vgg_16_demo')
    # ourModel.readModelFromDisk('unprotectedCifar10VGG.h5')
    # print(unprotectedModel.evaluate([np.expand_dims(testX, axis=3), np.expand_dims(np.zeros(testX.shape), axis=3)], tf.keras.utils.to_categorical(testY, numberOfClasses)))

    #re-store the model as a tensorflowv2 model
    ourModel.storeTensorflowV2('unprotectedCifar10VGG_TFV2')