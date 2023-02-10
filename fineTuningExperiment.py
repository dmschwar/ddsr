########################################################################################################################
#
# Author: David Schwartz, June, 9, 2020
#
# todo: args for batch based or "progressive" (i.e. generate adversarial samples once vs regenerate them in between fine tuning epochs)
########################################################################################################################
# imports

import gc
from DDSR import DDSR
from LDR_VGG_16 import LDRVGG
from HLRGD_VGG_16 import HLRGDVGG
import os, sys, time, copy
import argparse
from itertools import chain
import pickle
import cv2
import matplotlib.pyplot as plt
from adjustText import adjust_text
import ctypes
import sys
import multiprocessing #importing the module
multiprocessing.set_start_method('spawn')
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
# from numba import cuda 
# device = cuda.get_current_device()

import numpy as np
import tensorflow as tf

#detect OS to set temporary directory prefix
from sys import platform
if platform == "linux" or platform == "linux2":
    tmpPathPrefix = '/tmp/'
    #IF ON THE HPC
    # https://github.com/NVIDIA/framework-determinism/blob/master/doc/tensorflow.md
    # os.environ['PYTHONHASHSEED']=str(seed)
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # random.seed(a=seed, version=2)
    # tf.random.set_seed(seed)

elif platform == "win32":
    tmpPathPrefix = 'A:/tmp/'
    tf.config.experimental.enable_op_determinism
    

physical_devices = tf.config.list_physical_devices('GPU')
# os.environ['AUTOGRAPH_VERBOSITY'] = 1
os.environ['TF_GPU_THREAD_MODE']='gpu_private'
os.environ['TF_GPU_THREAD_COUNT']='1'
# os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
numberOfGPUs = len(physical_devices)
tf.print("Num GPUs Available: ", numberOfGPUs)
print("Num GPUs Available: ", numberOfGPUs)
try:
    for curGPU in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[curGPU], True)
except:
    pass
#if we have more than 1 gpu, distribute training across them all
if (numberOfGPUs > 1):
    distributeModels = True
    


import random
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from sklearn import model_selection
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import _safe_indexing as safe_indexing, indexable
########################################################################################################################

########################################################################################################################
#define our FGSM routine
loss_object = tf.keras.losses.CategoricalCrossentropy()
#@tf.function()
def create_adversarial_pattern(inputImageTensor, inputLabel, inputModel):
    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
        tape.watch(inputImageTensor)
        prediction = inputModel([inputImageTensor, inputImageTensor])
        loss = loss_object(inputLabel, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, inputImageTensor)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    del gradient, loss, prediction, inputImageTensor, inputLabel, inputModel, tape
    gc.collect()
    return signed_grad
########################################################################################################################


########################################################################################################################
#some training hyperparameters

chosenOptimizer = None#'SGD'
testingStability = False
defaultPatience = 50
defaultLossThreshold = 0.001
testBetweenEpochs = 1
singleAttack = True
#specify attack parameters
attackSpec = 'fgsm'
pgdSteps = 10

#set random seeds once
seed = 42
#tf.compat.v1.reset_default_graph()


#select the dataset
dataset = 'cifar10'#'fashion_mnist'
# naiveDDSRModelPath = 'explicitDataDependenceUnprotectedDDSR_3_12_22_bs_512_e_8192.h5'#'implicitDataDependenceUnprotectedDDSR.h5'#'explicitDataDependenceUnprotectedDDSR_3_12_22_bs_512_e_8192.h5'#'explicitDataDependenceUnprotectedDDSR.h5'

# define parameters as a function of the chosen dataset
if (dataset == 'fashion_mnist'):
    chosenReg = reg='KL'#'LDR'#'LDR'#'HLRlayer'#reg='LDR'
    naiveDDSRModelPath = naiveModelName = 'unprotectedFashionMNIST.h5'#'unprotectedFashionMNIST.h5'#'unprotectedCifar10VGG.h5'#'unprotectedFashionMNIST.h5'#'unprotectedCifar10VGG.h5'#'unprotectedFashionMNIST.h5'#"hlr_vgg16_1605695139430"

    verbose = True
    noisyTraining = True
    noisyTrainingSigma = 1./255
    numberOfAGNSamples = 1
    numberOfReplications = 10
    testFraction = 0.25
    numberOfClasses = 10
    kerasBatchSize = 16
    validationSplit =.15
    penaltyCoefficientLDR = 0.003
    penaltyCoefficientKL = 0.003
    penaltyCoefficientFIM = 0.003
    # trainingSetSize = 10000
    numberOfAdvSamplesTrain = 512
    numberOfAdvSamplesTest = 128
    trainingEpochs = 32
    dropoutRate = 0
    advPowers = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
    noisePowers = list(np.arange(0.05, 0.35, 0.1))
    trainingBudgetMax = 1./255
    adversarialMAEBudget = 8./255
    trainingBudgets = [trainingBudgetMax/5, trainingBudgetMax/4, trainingBudgetMax/2, trainingBudgetMax]
    trainingBudgets = np.insert(trainingBudgets, 0, 0, axis=0)#include benign samples in training data
    numberOfTrainingBudgets = len(trainingBudgets)
    print("training budgets:")
    print(trainingBudgets )

elif(dataset == 'cifar10'):
    exponentiallyDistributedEpsilons=10#10000
    chosenReg = None#'KLTimesL1'#'KL'#'KLTimesL1'#'KL'
    naiveDDSRModelPath = naiveModelName = 'DDSR_resonators_on_256.h5'#'explicitDataDependenceUnprotectedDDSR_with_resonatorsOn2048Epochs.h5'#'explicitDataDependenceUnprotectedDDSR_3_12_22_bs_512_e_8192.h5'#'explicitDataDependenceUnprotectedDDSR.h5'#'deeperVGGCifar.h5'#'unprotectedCifar10.h5'#old:'unprotectedCifar10VGG.h5'
    naiveNonResonatingModelPath = 'DDSR_resonators_off_256.h5'#'explicitDataDependenceUnprotectedDDSR_2048_resonatorsOff_bs512_5_17.h5'#'explicitDataDependenceUnprotectedDDSR_with_resonatorsOff1024Epochs.h5'#'explicitDataDependenceUnprotectedDDSR_with_resonatorsOff_128Epochs.h5'#'explicitDataDependenceUnprotectedDDSR_3_12_22_bs_512_e_8192.h5'#'explicitDataDependenceUnprotectedDDSR_with_resonatorsOff1024Epochs.h5'
    explicitDataDependence = True
    verbose = True
    noisyTraining = True
    multipleForwardPasses = None
    noisyTrainingSigma = 0.025#1./512
    numberOfAGNSamples = 1
    numberOfReplications = 10#5
    testFraction = 0.25
    numberOfClasses = 10
    numberOfPertSamples = kerasBatchSize = 16#512#256
    validationSplit = 0.25#.15#.25
    penaltyCoefficientKL = 0.0025#0.003
    penaltyCoefficientLDR = 0.0025
    penaltyCoefficientFIM = 0.0025
    eigenRegModelTrainingEpochs = 5
    trainingSetSize = 100000000
    numberOfAdvSamplesTrain = 512#768
    numberOfAdvSamplesTest = 256
    trainingEpochs = 16#was 16 11.1
    dropoutRate = 0
    advPowers = [0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
    noisePowers = list(np.arange(0.05, 0.35, 0.1))
    trainingBudgetMax = 32./255
    adversarialMAEBudget = 8./255
    trainingBudgets=[0.025]#1./512]#1./512, 1./256, #trainingBudgets = [0, 0, 0.00001, 0.001, 0.25]#[0, 1e-4, 1e-6, trainingBudgetMax/2, trainingBudgetMax]
    attackParams={'trainingEpsilon': 0.0125}#1./512}
    print(trainingBudgets)
    numberOfTrainingBudgets = len(trainingBudgets)
    print("training budgets:")
    print(trainingBudgets)


# input image dimensions
img_rows, img_cols = 32, 32#28, 28
########################################################################################################################

########################################################################################################################
# split data between official train and test sets
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

# trainX, testX, trainY, testY = copy.copy(trainX), copy.copy(testX), copy.copy(trainY), copy.copy(testY)
trainX, testX, trainY, testY = copy.copy(x_train), copy.copy(x_test), copy.copy(y_train), copy.copy(y_test)
stackedData = testX
stackedTargets = testY

#if we are only running one replication of the CV experiment
if (numberOfReplications == 1):
    splitIndices = [(range(stackedData.shape[0])[:np.floor(stackedData.shape[0]/2.).astype(int)], range(stackedData.shape[0])[(np.floor(stackedData.shape[0]/2.).astype(int)+1):])]
#if k > 1
else:
    kFold = StratifiedKFold(n_splits=numberOfReplications, shuffle=True, random_state=42)
    splitIndices = [(trainIndices, testIndices) for trainIndices, testIndices in kFold.split(stackedData, stackedTargets)]
########################################################################################################################

########################################################################################################################
#execute the cross val experiment
crossValResults = dict()
#store all arguments for reference later
argsDict = dict()
argsDict['chosenReg'] =  chosenReg
argsDict['naiveModelName'] = naiveModelName
argsDict['noisyTraining'] = noisyTraining
argsDict['noisyTrainingSigma'] = noisyTrainingSigma
argsDict['numberOfAGNSamples'] = numberOfAGNSamples
argsDict['numberOfReplications'] = numberOfReplications
argsDict['testFraction'] = testFraction
argsDict['numberOfClasses'] = numberOfClasses
argsDict['kerasBatchSize'] = kerasBatchSize
argsDict['kerasBatchSize'] = kerasBatchSize
argsDict['validationSplit'] = validationSplit
argsDict['penaltyCoefficientKL'] = penaltyCoefficientKL
argsDict['penaltyCoefficientLDR'] = penaltyCoefficientLDR
argsDict['penaltyCoefficientFIM'] = penaltyCoefficientFIM
argsDict['trainingSetSize'] = trainingSetSize
argsDict['numberOfAdvSamplesTrain'] = numberOfAdvSamplesTrain
argsDict['numberOfAdvSamplesTest'] = numberOfAdvSamplesTest 
argsDict['trainingEpochs']  = trainingEpochs
argsDict['dropoutRate']  = dropoutRate
argsDict['advPowers'] = advPowers
argsDict['noisePowers'] = noisePowers
argsDict['trainingBudgetMax'] = trainingBudgetMax
argsDict['adversarialMAEBudget'] = adversarialMAEBudget
argsDict['trainingBudgets'] = trainingBudgets
argsDict['explicitDataDependence'] = explicitDataDependence
argsDict['exponentiallyDistributedEpsilons'] = exponentiallyDistributedEpsilons
crossValResults['argsDict'] = argsDict
crossValResults['noisyTraining'] = noisyTraining
crossValResults['Acc'] = dict()
crossValResults['noiseAcc'] = dict()
crossValResults['splitIndices'] = dict()
crossValResults['averageLayerwiseUndefMAEs'] = dict()
crossValResults['averageLayerwiseLDRMAEs'] = dict()
crossValResults['averageLayerwiseKLMAEs'] = dict()
crossValResults['averageLayerwiseHLRGDMAEs'] = dict()
crossValResults['averageLayerwiseFIMMAEs'] = dict()
crossValResults['averageLayerwiseAGNMAEs'] = dict()
crossValResults['averageLayerwiseAGNUndefMAEs'] = dict()
crossValResults['averageLayerwiseAGNLDRMAEs'] = dict()
crossValResults['averageLayerwiseAGNHLRGDMAEs'] = dict()
crossValResults['averageLayerwiseAGNFIMMAEs'] = dict()
crossValResults['averageLayerwiseAGNAGNMAEs'] = dict()
crossValResults['averageLayerwiseAGNKLMAEs'] = dict()
crossValResults['budgets'] = advPowers
crossValResults['noiseBudgets'] = noisePowers
crossValResults['testBetweenEpochs'] = testBetweenEpochs
cvIndex = 0

for trainIndices, testIndices in splitIndices:

    #slice this iteration's data
    trainX, testX = stackedData[trainIndices, :, :, :], stackedData[testIndices, :, :, :]
    trainY, testY = stackedTargets[trainIndices, :], stackedTargets[testIndices, :]
    inputTestX = testX[:numberOfAdvSamplesTest, :, :, :]
    testYCat = tf.keras.utils.to_categorical(testY)[:numberOfAdvSamplesTest]

    print("%s total training indices"%str(len(trainIndices)))
    crossValResults['splitIndices'][cvIndex] = (trainIndices, testIndices)

    naiveModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0, optimizer=chosenOptimizer,
                            loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                            max_relu_bound=1.1, verbose=False, stochasticResonatorOn=True, explicitDataDependence=explicitDataDependence, usingAdvReg=False)
    naiveModel.model.load_weights(naiveModelName)#.expect_partial()
    naiveModelWeightsCopy = copy.copy(naiveModel.model.get_weights())
    
    naiveNonResonatingModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0, optimizer=chosenOptimizer,
                            loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                            max_relu_bound=1.1, verbose=False, stochasticResonatorOn=False, explicitDataDependence=explicitDataDependence, usingAdvReg=False)
    
    naiveNonResonatingModel.model.load_weights(naiveNonResonatingModelPath)
    naiveNonResonatingModelWeightsCopy = copy.copy(naiveNonResonatingModel.model.get_weights())
    
    print("unprotected model(s) loaded from disk")

    #set up containers to store perturbations
    fgsmData = dict()
    awgnData = dict()
    fgsmData['train'] = np.zeros((numberOfAdvSamplesTrain, 32, 32, 1))
    fgsmData['test'] = np.zeros((numberOfAdvSamplesTest, 32, 32, 1))
    fgsmData['trainResonatorsOff'] = np.zeros((numberOfAdvSamplesTrain, 32, 32, 1))
    fgsmData['trainResonatorsOff'] = np.zeros((numberOfAdvSamplesTest, 32, 32, 1))
    awgnData['test'] = dict()
    # generate awgn noise
    for i in range(len(noisePowers)):
        awgnData['test'][i] = np.random.normal(0, scale=noisePowers[i], size=testX.shape)
    inputTestAGNPerts = awgnData['test']
    print("awg noise generated")

    #attack the unprotected model
    fgsmData['train'] = K.eval(create_adversarial_pattern(tf.constant(trainX[:numberOfAdvSamplesTrain, :, : ,:]), tf.constant(tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], num_classes=numberOfClasses)), \
        inputModel=naiveNonResonatingModel.model))
    fgsmData['test'] = K.eval(create_adversarial_pattern(tf.constant(testX[:numberOfAdvSamplesTest, :, : ,:]), tf.constant(tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses)), \
        inputModel=naiveNonResonatingModel.model))   #if we're testing explicit data dependence (for stochastic resonators), then there are two naive models, and we must also attack both the naive model with resonators off and also the one with resonators enabled
    #in order to fairly compare the success of a defense method
    if (explicitDataDependence):
        fgsmData['trainResonatorsOff'] = K.eval(create_adversarial_pattern(tf.constant(trainX[:numberOfAdvSamplesTrain, :, : ,:]), tf.constant(tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], num_classes=numberOfClasses)),\
            inputModel=naiveNonResonatingModel.model))
        fgsmData['testResonatorsOff'] = K.eval(create_adversarial_pattern(tf.constant(testX[:numberOfAdvSamplesTest, :, : ,:]), tf.constant(tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses)), \
            inputModel=naiveNonResonatingModel.model))
    print("attack complete")

    #form the combined multibudget training set
    trainingAttacks, trainingAttacksResonatorsOff = np.array([]), np.array([])
    trainingBenigns = np.array([])
    distTrainingLabels = np.array([])
    for curTrainingBudget in trainingBudgets:
        if (exponentiallyDistributedEpsilons is not None):
            curTrainingAttacks = np.zeros_like(trainX[:numberOfAdvSamplesTrain, :, :, :])
            curTrainingAttacksResonatorsOff = np.zeros_like(trainX[:numberOfAdvSamplesTrain, :, :, :])
            curTrainingBudget = np.random.gamma(shape=0.25, scale=curTrainingBudget/exponentiallyDistributedEpsilons, size=numberOfAdvSamplesTrain)
            for subBudgetIndex in range(numberOfAdvSamplesTrain):
                curTrainingAttacks[subBudgetIndex,:,:,:] = trainX[subBudgetIndex, :, :, :] + curTrainingBudget[subBudgetIndex]*fgsmData['train'][subBudgetIndex,:,:,:]    
                if (explicitDataDependence):
                    curTrainingAttacksResonatorsOff[subBudgetIndex,:,:,:] = trainX[subBudgetIndex, :, :, :] + curTrainingBudget[subBudgetIndex]*fgsmData['trainResonatorsOff'][subBudgetIndex,:,:,:]    
        else:
            curTrainingAttacks = trainX[:numberOfAdvSamplesTrain, :, :, :] + curTrainingBudget*fgsmData['train']
        trainingAttacks = np.concatenate((trainingAttacks, curTrainingAttacks), axis=0) if trainingAttacks.size > 0 else curTrainingAttacks
        if (explicitDataDependence):
            trainingAttacksResonatorsOff = np.concatenate((trainingAttacksResonatorsOff, curTrainingAttacksResonatorsOff), axis=0) if trainingAttacksResonatorsOff.size > 0 else curTrainingAttacksResonatorsOff
        trainingBenigns = np.concatenate((trainingBenigns, trainX[:numberOfAdvSamplesTrain, :, :, :]), axis=0) if trainingBenigns.size > 0 else trainX[:numberOfAdvSamplesTrain, :, :, :]
        distTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), distTrainingLabels), axis=0) if distTrainingLabels.size > 0 \
            else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)

    #this block includes gaussianly noisy data to be included in the attack component of this training set
    if (noisyTraining):
        noiseOnlyTrainingAttacks, noiseOnlyTrainingBenigns, noiseOnlyTrainingLabels = np.array([]), np.array([]), np.array([])
        for curAGNSample in range(numberOfAGNSamples):
            curNoisyTrainingSamples = trainX[:numberOfAdvSamplesTrain, :, :, :] + np.random.normal(0, scale=noisyTrainingSigma, size=(numberOfAdvSamplesTrain, trainX.shape[1], trainX.shape[2], trainX.shape[3]))

            noiseOnlyTrainingAttacks = np.concatenate((noiseOnlyTrainingAttacks, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingAttacks.size > 0 else curNoisyTrainingSamples
            noiseOnlyTrainingBenigns = np.concatenate((noiseOnlyTrainingBenigns, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingBenigns.size > 0 else curNoisyTrainingSamples
            noiseOnlyTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), noiseOnlyTrainingLabels), axis=0) if noiseOnlyTrainingLabels.size > 0 \
                else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)
            noiseOnlyTrainingAttacks = np.concatenate((noiseOnlyTrainingAttacks, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingAttacks.size > 0 else curNoisyTrainingSamples
            noiseOnlyTrainingBenigns = np.concatenate((noiseOnlyTrainingBenigns, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingBenigns.size > 0 else curNoisyTrainingSamples
            noiseOnlyTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), noiseOnlyTrainingLabels), axis=0) if noiseOnlyTrainingLabels.size > 0 \
                else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)
            #form exclusive awgn training data twice
            for z in range(numberOfTrainingBudgets):
                noiseOnlyTrainingAttacks = np.concatenate((noiseOnlyTrainingAttacks, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingAttacks.size > 0 else curNoisyTrainingSamples
                noiseOnlyTrainingBenigns = np.concatenate((noiseOnlyTrainingBenigns, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingBenigns.size > 0 else curNoisyTrainingSamples
                noiseOnlyTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), noiseOnlyTrainingLabels), axis=0) if noiseOnlyTrainingLabels.size > 0 \
                    else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)

                # noiseOnlyTrainingAttacks = np.concatenate((noiseOnlyTrainingAttacks, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingAttacks.size > 0 else curNoisyTrainingSamples
                # noiseOnlyTrainingBenigns = np.concatenate((noiseOnlyTrainingBenigns, curNoisyTrainingSamples), axis=0) if noiseOnlyTrainingBenigns.size > 0 else curNoisyTrainingSamples
                # noiseOnlyTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), noiseOnlyTrainingLabels), axis=0) if noiseOnlyTrainingLabels.size > 0 \
                #    else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)

            #append disparity regularized noisy training data to other models adv training sets#
            trainingAttacks = np.concatenate((trainingAttacks, curNoisyTrainingSamples), axis=0) if trainingAttacks.size > 0 else curNoisyTrainingSamples
            trainingBenigns = np.concatenate((trainingBenigns, trainX[:numberOfAdvSamplesTrain, :, :, :]), axis=0) if trainingBenigns.size > 0 else curNoisyTrainingSamples
            distTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), distTrainingLabels), axis=0) if distTrainingLabels.size > 0 \
            else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)

            # #append disparity unregularized noisy training data to other models adv training sets
            trainingAttacks = np.concatenate((trainingAttacks, curNoisyTrainingSamples), axis=0) if trainingAttacks.size > 0 else curNoisyTrainingSamples
            trainingBenigns = np.concatenate((trainingBenigns, curNoisyTrainingSamples), axis=0) if trainingBenigns.size > 0 else curNoisyTrainingSamples
            distTrainingLabels = np.concatenate((tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses), distTrainingLabels), axis=0) if distTrainingLabels.size > 0 \
                else tf.keras.utils.to_categorical(trainY[:numberOfAdvSamplesTrain], numberOfClasses)

    #print fine tuning dataset dimensionality and size
    if (noisyTraining):
        print("%s gaussianly noisy training samples generated" % str(noiseOnlyTrainingAttacks.shape[0]))
    print("%s adv. training samples generated"%str(trainingAttacks.shape[0]))
    testingAttacks = inputTestX + np.max(trainingBudgets) * fgsmData['test']
    print("testing attacks generated")

    #split the data into training and validation subsets
    preSplitData, preSplitTargets = [trainingBenigns, trainingAttacks], distTrainingLabels
    if (explicitDataDependence):
        preSplitDataResonatorsOff = [trainingBenigns, trainingAttacksResonatorsOff]
    preSplitNoiseData = [noiseOnlyTrainingAttacks, noiseOnlyTrainingAttacks]
    if (validationSplit > 0):
        #split the data (here, so we don't risk having slightly different training sets for different models) for validation
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validationSplit, random_state=42)
        arrays = indexable(preSplitData[0], preSplitTargets)
        train, test = next(sss.split(X=preSplitData[0], y=preSplitTargets))
        iterator = list(chain.from_iterable((safe_indexing(a, train),
                                                safe_indexing(a, test),
                                                train,
                                                test) for a in arrays))
        X_train, X_val, train_is, val_is, y_train, y_val, _, _ = iterator
        trainingData, validationData = [], []
        trainingData = [preSplitData[0][train_is,:,:], preSplitData[1][train_is, :, :]]
        validationData = [preSplitData[0][val_is,:,:], preSplitData[1][val_is, :, :]]
        awgnTrainingData = [preSplitNoiseData[0][train_is,:,:], preSplitNoiseData[1][train_is, :, :]]
        awgnValidationData = [preSplitNoiseData[0][val_is,:,:], preSplitNoiseData[1][val_is, :, :]]
        #adv. versions of the above lists allow for adversarial training (not just adv-regularized benign training))
        advTrainingData = [preSplitData[1][train_is,:,:], preSplitData[0][train_is, :, :]]
        advValidationData = [preSplitData[1][val_is,:,:], preSplitData[0][val_is, :, :]]
        trainingTargets, validationTargets = y_train, y_val
        
        #if we have models with resonators on (and therefore are comparing to models with resonators off) then we need separate adv validation data for those defending naive models with the resonators off
        if (explicitDataDependence):
            advTrainingDataResonatorsOff = [preSplitData[1][train_is,:,:], preSplitData[0][train_is, :, :]]
            advValidationDataResonatorsOff = [preSplitData[1][val_is,:,:], preSplitData[0][val_is, :, :]]


    else:
        print("set validation split fraction (validationSplit)")
    
    print("Data are split")



    #set up containers for results
    advAccuracy, noiseAccuracy, allAdvAccuracies, allNoiseAccuracies = dict(), dict(), dict(), dict()
    
    #test the undefended model first
    benignResultsUnprotected = naiveNonResonatingModel.evaluate([inputTestX, inputTestX], testYCat, batchSize=kerasBatchSize)
        
    advAccuracy['undefended'], advAccuracy['LDR'], advAccuracy['ddsr'], advAccuracy['fim'], advAccuracy['hlrgd'], advAccuracy['agn'], advAccuracy['ddsr_MFP'] = \
        [benignResultsUnprotected[1]], [], [], [], [], [], []
    noiseAccuracy['undefended'], noiseAccuracy['LDR'], noiseAccuracy['ddsr'], noiseAccuracy['fim'], noiseAccuracy['hlrgd'], noiseAccuracy['agn'], noiseAccuracy['ddsr_MFP']= \
        [benignResultsUnprotected[1]], [], [], [], [], [], []

    # if (explicitDataDependence):

    
    #untrained adv noise test
    for curNoisePower in advPowers:
        #calculate new testingAttacks
        testingAttacks = inputTestX + curNoisePower*fgsmData['test']
        testingAttacksResonatorsOff = inputTestX + curNoisePower*fgsmData['testResonatorsOff']
        #get performance
        unprotectedResult = naiveNonResonatingModel.evaluate([testingAttacksResonatorsOff, testingAttacksResonatorsOff], testYCat, batchSize=kerasBatchSize)[1]
        advAccuracy['undefended'].append(unprotectedResult)
    #untrained awg noise test
    for curNoiseIndex in range(len(noisePowers)):
        #calculate new testingAttacks
        corruptedTestX = inputTestX + awgnData['test'][curNoiseIndex][:numberOfAdvSamplesTest, :, :, :]
        #get performance
        unprotectedResult = naiveNonResonatingModel.evaluate([corruptedTestX, corruptedTestX], testYCat, batchSize=kerasBatchSize)[1]
        noiseAccuracy['undefended'].append(unprotectedResult)


    #train an KL protected model (ours)
    ddsrProtectedModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0.05,\
        loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0,#0.33,
        max_relu_bound=1.1, verbose=False, usingAdvReg=False, reg=None, explicitDataDependence=explicitDataDependence, stochasticResonatorOn=True)
    if (multipleForwardPasses is not None):
        multipleForwardPassesModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0.05,\
            loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0,#0.33,
            max_relu_bound=1.1, verbose=False, usingAdvReg=False, reg=None, explicitDataDependence=explicitDataDependence, stochasticResonatorOn=True)
        ddsrProtectedModel.model.load_weights(naiveDDSRModelPath, by_name=False)
        multipleForwardPassesModel.model.load_weights(naiveDDSRModelPath, by_name=False)
    #define a training function wrapper
    # def ddsrTrainingWrapper():
    # scriptifiedTrainOnline = scriptifier.run_as_script(ddsrProtectedModel.trainOnline)
    # loss, acc, bestKLProtectedModelPath, benignResultsKL, noiseAccuracy['ddsr'], advAccuracy['ddsr'] = scriptifiedTrainOnline(trainingData[0], trainingTargets, #advTrainingData
    #                                                         advValidationData, validationTargets, training_epochs=trainingEpochs,
    #                                                         monitor='val_loss', patience=defaultPatience, model_path=None, #adversarialAttackFunction=create_adversarial_pattern,
    #                                                         keras_batch_size=kerasBatchSize, testBetweenEpochs=testBetweenEpochs, 
    #                                                         inputTestingData=inputTestX, inputTestingTargets=testYCat, 
    #                                                         inputTestingAGNPerturbations=inputTestAGNPerts,
    #                                                         inputAdvPerturbations=fgsmData['test'], noisePowers=noisePowers, 
    #                                                         advPowers=advPowers, dataAugmentation=False, 
    #                                                         adversarialOrder=1, normed=True, attackParams=attackParams, onlineAttackType="FGSM",
    #                                                         trainOnAWGNPerts=True, awgnPertsOnly=False, noisyTrainingSigma=noisyTrainingSigma)

        # return loss, acc, bestKLProtectedModelPath, benignResultsKL, noiseAccuracy['ddsr'], advAccuracy['ddsr']
    if (multipleForwardPasses is not None):
        loss, acc, bestKLProtectedModelPath, benignResultsMFP, noiseAccuracy['ddsr_MFP'], advAccuracy[
            'ddsr_MFP'], advAccuracy['ddsr_singleFP'],noiseAccuracy['ddsr_singleFP'], benignResultsKL = ddsrProtectedModel.trainOnline(trainingData[0], trainingTargets,  # advTrainingData
                                                     advValidationData, validationTargets,
                                                     training_epochs=trainingEpochs,
                                                     # adversarialAttackFunction=create_adversarial_pattern,
                                                     monitor='val_loss', patience=defaultPatience, model_path=None,
                                                     keras_batch_size=kerasBatchSize,
                                                     testBetweenEpochs=testBetweenEpochs,
                                                     inputTestingData=inputTestX, inputTestingTargets=testYCat,
                                                     inputTestingAGNPerturbations=inputTestAGNPerts,
                                                     inputAdvPerturbations=fgsmData['test'], noisePowers=noisePowers,
                                                     advPowers=advPowers, dataAugmentation=False,
                                                     adversarialOrder=1, normed=True, attackParams=attackParams,
                                                     onlineAttackType="FGSM",
                                                     trainOnAWGNPerts=True, awgnPertsOnly=False,
                                                     noisyTrainingSigma=noisyTrainingSigma,
                                                     multipleForwardPasses = multipleForwardPasses,
                                                     exponentiallyDistributedEpsilons=exponentiallyDistributedEpsilons,
                                                     singleAttack=singleAttack)
    else:
        loss, acc, bestKLProtectedModelPath, benignResultsKL, noiseAccuracy['ddsr'], advAccuracy[
            'ddsr'] = ddsrProtectedModel.trainOnline(trainingData[0], trainingTargets,  # advTrainingData
                                                     advValidationData, validationTargets,
                                                     training_epochs=trainingEpochs,
                                                     # adversarialAttackFunction=create_adversarial_pattern,
                                                     monitor='val_loss', patience=defaultPatience, model_path=None,
                                                     keras_batch_size=kerasBatchSize,
                                                     testBetweenEpochs=testBetweenEpochs,
                                                     inputTestingData=inputTestX, inputTestingTargets=testYCat,
                                                     inputTestingAGNPerturbations=inputTestAGNPerts,
                                                     inputAdvPerturbations=fgsmData['test'], noisePowers=noisePowers,
                                                     advPowers=advPowers, dataAugmentation=False,
                                                     adversarialOrder=1, normed=True, attackParams=attackParams,
                                                     onlineAttackType="FGSM",
                                                     trainOnAWGNPerts=True, awgnPertsOnly=False,
                                                     noisyTrainingSigma=noisyTrainingSigma,
                                                     exponentiallyDistributedEpsilons=exponentiallyDistributedEpsilons,
                                                     singleAttack=singleAttack)

    # prc1 = multiprocessing.Process(target=ddsrTrainingWrapper)
    # prc1.start()
    # prc1.join()
    # loss, acc, bestKLProtectedModelPath, benignResultsKL, noiseAccuracy['ddsr'], advAccuracy['ddsr'] = prc1.join()
    #clear backend to make room for the gradients for the next model's adversary
    ddsrProtectedModel.deleteModels()
    del ddsrProtectedModel
    #tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    print(gc.collect())
    
    #train an LDR protected model (ours)
    l1ProtectedModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=penaltyCoefficientLDR,
                            loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                            max_relu_bound=1.1, reg='LDR', stochasticResonatorOn=False, explicitDataDependence=explicitDataDependence, usingAdvReg=True, verbose=False)
    # l1ProtectedModel.model.set_weights(naiveModelWeightsCopy)
    l1ProtectedModel.model.load_weights(naiveNonResonatingModelPath, by_name=False)
    loss, acc, bestL1ProtectedModelPath, benignResultsLDR, noiseAccuracy['LDR'], advAccuracy['LDR'] = l1ProtectedModel.trainOnline(trainingData[0], trainingTargets,
                                                            advValidationData, validationTargets, training_epochs=trainingEpochs,#adversarialAttackFunction=create_adversarial_pattern,
                                                            monitor='val_loss', patience=defaultPatience, model_path=None, 
                                                            keras_batch_size=kerasBatchSize, testBetweenEpochs=testBetweenEpochs, 
                                                            inputTestingData=inputTestX, inputTestingTargets=testYCat, 
                                                            inputTestingAGNPerturbations=inputTestAGNPerts,
                                                            inputAdvPerturbations=fgsmData['test'] if (not explicitDataDependence) else fgsmData['testResonatorsOff'], noisePowers=noisePowers, 
                                                            advPowers=advPowers, dataAugmentation=False, 
                                                            adversarialOrder=1, normed=True, attackParams=attackParams, onlineAttackType="FGSM",
                                                            trainOnAWGNPerts=True, awgnPertsOnly=False, noisyTrainingSigma=noisyTrainingSigma, 
                                                            exponentiallyDistributedEpsilons=exponentiallyDistributedEpsilons, singleAttack=singleAttack)
    #clear backend to make room for the gradients for the next model's adversary
    l1ProtectedModel.deleteModels()
    del l1ProtectedModel
    #tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    print(gc.collect())


    #train an FIM regularization protected model (Shen's)
    #safe to give benign training set to this because this model ignores the second component in the loss calculation
    eigenRegModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=penaltyCoefficientFIM,
                            loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                            max_relu_bound=1.1, reg='FIM', usingAdvReg=True, explicitDataDependence=explicitDataDependence, stochasticResonatorOn=False, verbose=False)
    eigenRegModel.model.load_weights(naiveNonResonatingModelPath, by_name=False)
    eigenRegLoss, acc, bestEigenRegModelPath, benignResultsFIM, noiseAccuracy['fim'], advAccuracy['fim'] = eigenRegModel.trainOnline(trainingData[0], trainingTargets, validationData, validationTargets,
                                                        training_epochs=eigenRegModelTrainingEpochs, monitor='val_loss', patience=defaultPatience, model_path=None,#adversarialAttackFunction=create_adversarial_pattern,
                                                        keras_batch_size=kerasBatchSize, testBetweenEpochs=testBetweenEpochs, 
                                                        inputTestingData=inputTestX, inputTestingTargets=testYCat, 
                                                        inputTestingAGNPerturbations=inputTestAGNPerts,
                                                        inputAdvPerturbations=fgsmData['test'] if (not explicitDataDependence) else fgsmData['testResonatorsOff'], noisePowers=noisePowers, 
                                                        advPowers=advPowers, dataAugmentation=False, 
                                                        adversarialOrder=0, normed=True, attackParams=attackParams, onlineAttackType="FGSM",
                                                        trainOnAWGNPerts=False, awgnPertsOnly=False, noisyTrainingSigma=noisyTrainingSigma, 
                                                        exponentiallyDistributedEpsilons=exponentiallyDistributedEpsilons, singleAttack=singleAttack)
    #clear backend to make room for the gradients for the next model's adversary
    eigenRegModel.deleteModels()
    del eigenRegModel
    #tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    print(gc.collect())


    benignResultsAGN = [-1, 1]
    #train the AWGN model
    if (noisyTraining):
        awgnModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0, explicitDataDependence=explicitDataDependence,
                                loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                                max_relu_bound=1.1, verbose=False, stochasticResonatorOn=False, usingAdvReg=False)
        #train the awgn only model
        awgnModel.model.set_weights(naiveModelWeightsCopy)
        loss, acc, bestawgnModelPath, benignResultsAGN, noiseAccuracy['agn'], advAccuracy['agn'] = awgnModel.trainOnline(trainingData[0], trainingTargets, awgnValidationData, validationTargets,
                                                        training_epochs=trainingEpochs, monitor='val_loss', patience=defaultPatience, model_path=None,#adversarialAttackFunction=create_adversarial_pattern,
                                                        keras_batch_size=kerasBatchSize, testBetweenEpochs=testBetweenEpochs, 
                                                        inputTestingData=inputTestX, inputTestingTargets=testYCat, 
                                                        inputTestingAGNPerturbations=inputTestAGNPerts,
                                                        inputAdvPerturbations=fgsmData['test'] if (not explicitDataDependence) else fgsmData['testResonatorsOff'], noisePowers=noisePowers, 
                                                        advPowers=advPowers, dataAugmentation=False, 
                                                        adversarialOrder=1, normed=True, attackParams=attackParams, onlineAttackType=None,
                                                        trainOnAWGNPerts=True, awgnPertsOnly=True, noisyTrainingSigma=noisyTrainingSigma, 
                                                        exponentiallyDistributedEpsilons=exponentiallyDistributedEpsilons, singleAttack=singleAttack)
        #clear backend to make room for the gradients for the next model's adversary
        awgnModel.deleteModels()
        del awgnModel
        #tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()           
        print(gc.collect())

    hlrgd = HLRGDVGG(input_shape, numberOfClasses, vggModelLocation=None,#os.path.join(os.getcwd(),naiveModelName),
                        number_of_classes=numberOfClasses, adv_penalty=0.0005, explicitDataDependence=explicitDataDependence,
                        loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                        max_relu_bound=1.1, verbose=False)
    #set underlying model weights
    hlrgd.ourVGG.model.load_weights(naiveNonResonatingModelPath, by_name=False)
    #train an hlrgd (no need to load the weights here because the hlrgd train() routine does this at completion)
    loss, acc, bestHLRGDModelPath, benignResultsHLRGD, noiseAccuracy['hlrgd'], advAccuracy['hlrgd'] = hlrgd.train(trainingData, trainingTargets, validationData, validationTargets,
                                                        training_epochs=trainingEpochs, monitor='val_loss', patience=defaultPatience, model_path=None, 
                                                        keras_batch_size=kerasBatchSize, testBetweenEpochs=testBetweenEpochs, 
                                                        inputTestingData=inputTestX, inputTestingTargets=testYCat, 
                                                        inputTestingAGNPerturbations=inputTestAGNPerts,
                                                        inputAdvPerturbations=fgsmData['test'] if (not explicitDataDependence) else fgsmData['testResonatorsOff'], noisePowers=noisePowers, 
                                                        advPowers=advPowers, dataAugmentation=False, 
                                                        adversarialOrder=0, normed=True)
    #clear backend to make room for the gradients for the next model's adversary
    hlrgd.deleteModels()
    del hlrgd
    #tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    print(gc.collect())



    #now that all models are trained, reinstantiate and read in their weights
    naiveModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0, optimizer=chosenOptimizer,
                        loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                        max_relu_bound=1.1, verbose=False, stochasticResonatorOn=False, explicitDataDependence=explicitDataDependence, usingAdvReg=False)
    naiveModel.model.load_weights(naiveModelName) 
    
    naiveNonResonatingModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0, optimizer=chosenOptimizer,
                                loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                                max_relu_bound=1.1, verbose=False, stochasticResonatorOn=False, explicitDataDependence=explicitDataDependence, usingAdvReg=False)
        
    naiveNonResonatingModel.model.load_weights(naiveNonResonatingModelPath)
    
    ddsrProtectedModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0.05,
                    loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=0,#0.33,
                    max_relu_bound=1.1, verbose=False, usingAdvReg=True, reg='LDR', explicitDataDependence=explicitDataDependence, stochasticResonatorOn=True)
    ddsrProtectedModel.model.load_weights(bestKLProtectedModelPath)

    eigenRegModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=penaltyCoefficientFIM,
                            loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                            max_relu_bound=1.1, reg='FIM', usingAdvReg=True, explicitDataDependence=explicitDataDependence, stochasticResonatorOn=False, verbose=False)
    if (not (np.isnan(eigenRegLoss).any() or np.isinf(eigenRegLoss).any())):
        eigenRegModel.model.load_weights(bestEigenRegModelPath)
    else:
        eigenRegModel.model.load_weights(naiveModelWeightsCopy)
    l1ProtectedModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=penaltyCoefficientLDR,
                                    loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                                    max_relu_bound=1.1, reg='LDR', stochasticResonatorOn=False, explicitDataDependence=explicitDataDependence, usingAdvReg=True, verbose=False)
    l1ProtectedModel.model.load_weights(bestL1ProtectedModelPath)

    if (noisyTraining):
        awgnModel = DDSR(input_shape, numberOfClasses, number_of_classes=numberOfClasses, adv_penalty=0, explicitDataDependence=explicitDataDependence,
                                loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                                max_relu_bound=1.1, verbose=False, stochasticResonatorOn=False, usingAdvReg=False)        
        awgnModel.model.load_weights(bestawgnModelPath)
    
    hlrgd = HLRGDVGG(input_shape, numberOfClasses, vggModelLocation=None,#os.path.join(os.getcwd(),naiveModelName),
                        number_of_classes=numberOfClasses, adv_penalty=0.0005, explicitDataDependence=explicitDataDependence,
                        loss_threshold=defaultLossThreshold, patience=defaultPatience, dropout_rate=dropoutRate,
                        max_relu_bound=1.1, verbose=False)
    hlrgd.ourVGG.model.set_weights(naiveModelWeightsCopy)
    hlrgd.model.load_weights(bestHLRGDModelPath)
    
    #convert acc results into numpy arrays for easy arithmetic
    if (not bool(allAdvAccuracies)):
        allAdvAccuracies['undefended'] = np.array(advAccuracy['undefended'])
        allAdvAccuracies['LDR'] = np.array(advAccuracy['LDR'])
        allAdvAccuracies['hlrgd'] = np.array(advAccuracy['hlrgd'])
        allAdvAccuracies['fim'] = np.array(advAccuracy['fim'])
        allNoiseAccuracies['undefended'] = np.array(noiseAccuracy['undefended'])
        allNoiseAccuracies['LDR'] = np.array(noiseAccuracy['LDR'])
        allNoiseAccuracies['hlrgd'] = np.array(noiseAccuracy['hlrgd'])
        allNoiseAccuracies['fim'] = np.array(noiseAccuracy['fim'])

        if multipleForwardPasses is not None:
            allAdvAccuracies['ddsr'] = np.array(advAccuracy['ddsr_singleFP'])
            allNoiseAccuracies['ddsr'] = np.array(noiseAccuracy['ddsr_singleFP'])
            allAdvAccuracies['ddsr_MFP'] = np.array(advAccuracy['ddsr_MFP'])
            allNoiseAccuracies['ddsr_MFP'] = np.array(noiseAccuracy['ddsr_MFP'])
        else:
            allAdvAccuracies['ddsr'] = np.array(advAccuracy['ddsr'])
            allNoiseAccuracies['ddsr'] = np.array(noiseAccuracy['ddsr'])


        # noiseAccuracy['ddsr_MFP'], advAccuracy[
        #     'ddsr_MFP'], noiseAccuracy['ddsr_singleFP'], advAccuracy['ddsr_singleFP']
        if (noisyTraining):
            allAdvAccuracies['agn'] = np.array(advAccuracy['agn'])
            allNoiseAccuracies['agn'] = np.array(noiseAccuracy['agn'])
            
    else:
        allAdvAccuracies['undefended'] = np.concatenate((allAdvAccuracies['undefended'], np.array(advAccuracy['undefended'])), axis=0)
        allAdvAccuracies['LDR'] = np.concatenate((allAdvAccuracies['LDR'], np.array(advAccuracy['LDR'])), axis=0)

        allAdvAccuracies['hlrgd'] = np.concatenate((allAdvAccuracies['hlrgd'], np.array(advAccuracy['hlrgd'])), axis=0)
        allAdvAccuracies['fim'] = np.concatenate((allAdvAccuracies['fim'], np.array(advAccuracy['fim'])), axis=0)
        allNoiseAccuracies['undefended'] = np.concatenate((allNoiseAccuracies['undefended'], np.array(noiseAccuracy['undefended'])), axis=0)
        allNoiseAccuracies['LDR'] = np.concatenate((allNoiseAccuracies['LDR'], np.array(noiseAccuracy['LDR'])), axis=0)
        allNoiseAccuracies['hlrgd'] = np.concatenate((allNoiseAccuracies['hlrgd'], np.array(noiseAccuracy['hlrgd'])), axis=0)
        allNoiseAccuracies['fim'] = np.concatenate((allNoiseAccuracies['fim'], np.array(noiseAccuracy['fim'])), axis=0)

        if multipleForwardPasses is not None:
            allAdvAccuracies['ddsr'] = np.concatenate((allAdvAccuracies['ddsr_singleFP'], np.array(advAccuracy['ddsr_singleFP'])), axis=0)
            allNoiseAccuracies['ddsr'] = np.concatenate((allNoiseAccuracies['ddsr_singleFP'], np.array(noiseAccuracy['ddsr_singleFP'])), axis=0)
            allAdvAccuracies['ddsr_MFP'] = np.concatenate((allAdvAccuracies['ddsr_MFP'], np.array(advAccuracy['ddsr_MFP'])), axis=0)
            allNoiseAccuracies['ddsr_MFP'] = np.concatenate((allNoiseAccuracies['ddsr_MFP'], np.array(noiseAccuracy['ddsr_MFP'])), axis=0)
        else:
            allAdvAccuracies['ddsr'] = np.concatenate((allAdvAccuracies['ddsr'], np.array(advAccuracy['ddsr'])), axis=0)
            allNoiseAccuracies['ddsr'] = np.concatenate((allNoiseAccuracies['ddsr'], np.array(noiseAccuracy['ddsr'])), axis=0)

        if (noisyTraining):
            allAdvAccuracies['agn'] = np.concatenate((allAdvAccuracies['agn'], np.array(advAccuracy['agn'])), axis=0)
            allNoiseAccuracies['agn'] = np.concatenate((allNoiseAccuracies['agn'], np.array(noiseAccuracy['agn'])), axis=0)

    #print summary once after iterating over entire elongated training/testing process
    print("unprotected model")
    #evaluate the unprotected model
    attackResults = naiveNonResonatingModel.evaluate([testingAttacksResonatorsOff, testingAttacksResonatorsOff], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), batchSize=kerasBatchSize)
    print(attackResults)
    # benignResultsUnprotected = naiveModel.evaluate([inputTestX, inputTestX], testYCat, batchSize=kerasBatchSize)
    print(benignResultsUnprotected)

    print("protected eigenreg model")
    #evaluate the protected model
    attackResults = eigenRegModel.evaluate([testingAttacksResonatorsOff, testingAttacksResonatorsOff], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), batchSize=kerasBatchSize)
    print(attackResults) 
    # benignResultsFIM = eigenRegModel.evaluate([inputTestX, inputTestX], testYCat, batchSize=kerasBatchSize)
    print(benignResultsFIM)

    print("protected LDR")
    #evaluate the protected model
    attackResults = l1ProtectedModel.evaluate([testingAttacksResonatorsOff, testingAttacksResonatorsOff], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), batchSize=kerasBatchSize)
    print(attackResults)
    # benignResultsLDR = ddsrProtectedModel.evaluate([inputTestX, inputTestX], testYCat, batchSize=kerasBatchSize)
    print(benignResultsLDR)

    print("protected ddsr")
    #evaluate the protected model
    attackResults = ddsrProtectedModel.evaluate([testingAttacks, testingAttacks], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), batchSize=kerasBatchSize)
    print(attackResults)
    # benignResultsLDR = ddsrProtectedModel.evaluate([inputTestX, inputTestX], testYCat, batchSize=kerasBatchSize)
    print(benignResultsKL)

    if multipleForwardPasses is not None:
        print("protected ddsr Multiple Forward Passes")
        #evaluate the protected model
        attackResults = ddsrProtectedModel.evalMultipleForwardPasses([testingAttacks, testingAttacks], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), multipleForwardPasses, batch_size=kerasBatchSize)
        print(attackResults)
        # benignResultsLDR = ddsrProtectedModel.evaluate([inputTestX, inputTestX], testYCat, batchSize=kerasBatchSize)
        print(benignResultsMFP)


    if (noisyTraining):
        print("awgn")
        #evaluate the awgn model
        attackResultsAGN = awgnModel.evaluate([testingAttacksResonatorsOff, testingAttacksResonatorsOff], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), batchSize=kerasBatchSize)
        print(attackResultsAGN)
        # benignResultsAGN = awgnModel.evaluate([inputTestX, inputTestX], testYCat, batchSize=kerasBatchSize)
        print(benignResultsAGN)

    print("hlrgd")
    attackResults = hlrgd.evaluate([testingAttacksResonatorsOff, testingAttacksResonatorsOff], tf.keras.utils.to_categorical(testY[:numberOfAdvSamplesTest], num_classes=numberOfClasses), batchSize=kerasBatchSize)
    print(attackResults)
    # benignResultsHLRGD = hlrgd.evaluate([inputTestX, inputTestX], testYCat, batchSize=kerasBatchSize)
    print(benignResultsHLRGD)

    #set up containers to store perturbations
    # # fgsmData['test'] = np.zeros((numberOfAdvSamplesTest, input_shape[0], input_shape[1], input_shape[2]))
    # averageLayerwiseHLRGDMAEs, averageLayerwiseLDRMAEs, averageLayerwiseKLMAEs, averageLayerwiseUndefMAEs, averageLayerwiseFIMMAEs = dict(), dict(), dict(), dict()
    # averageLayerwiseAGNHLRGDMAEs, averageLayerwiseAGNLDRMAEs, averageLayerwiseAGNKLMAEs, averageLayerwiseAGNUndefMAEs, averageLayerwiseAGNFIMMAEs = dict(), dict(), dict(), dict()



    #recalculate the noisy test data
    corruptedTestX = inputTestX + np.random.normal(0, scale=np.sqrt(adversarialMAEBudget), size=(np.min([numberOfAdvSamplesTest, testX.shape[0]]), testX.shape[1], testX.shape[2], testX.shape[3]))
    
    # calculate awgn induced perturbations (only for the largest budget, which is currently curNoisePower
    curAdversarialPreds = l1ProtectedModel.hiddenModel.predict([corruptedTestX[:32, :, :, :], corruptedTestX[:32, :, :, :]])
    curAdversarialPredsDDSR = ddsrProtectedModel.hiddenModel.predict([corruptedTestX[:32, :, :, :], corruptedTestX[:32, :, :, :]])
    curAdversarialPredsHLRGD = hlrgd.ourVGG.hiddenModel.predict([corruptedTestX[:32, :, :, :], corruptedTestX[:32, :, :, :]])
    curAdversarialPredsUndef = naiveModel.hiddenModel.predict([corruptedTestX[:32, :, :, :], corruptedTestX[:32, :, :, :]])
    curAdversarialPredsFIM = eigenRegModel.hiddenModel.predict([corruptedTestX[:32, :, :, :], corruptedTestX[:32, :, :, :]])
    if (noisyTraining):
        curAdversarialPredsAGN = awgnModel.hiddenModel.predict([corruptedTestX[:32, :, :, :], corruptedTestX[:32, :, :, :]])
    #containers to store layerwise perturbational errors

    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)

    layerWiseAGNKLMAEs, layerWiseAGNLDRMAEs, layerWiseAGNHLRGDMAEs, layerWiseAGNUndefMAEs, layerWiseAGNFIMMAEs, layerWiseAGNAGNMAEs = [], [], [], [], [], []
    for curLayer in list(range(len(curAdversarialPreds)))[1::2]:
        curAdversarialMAEs = []
        LDRMAE = np.mean(K.eval(cosine_loss(curAdversarialPreds[curLayer],curAdversarialPreds[curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPreds[curLayer] - curAdversarialPreds[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPreds[curLayer+1]), axis=(1,2,3))), axis=0)
        KLMAE = np.mean(K.eval(cosine_loss(curAdversarialPredsDDSR[curLayer],curAdversarialPredsDDSR[curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPredsDDSR[curLayer] - curAdversarialPredsDDSR[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsDDSR[curLayer+1]), axis=(1,2,3))), axis=0)
        hlrgdMAE = np.mean(K.eval(cosine_loss(curAdversarialPredsHLRGD[curLayer],curAdversarialPredsHLRGD[curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPredsHLRGD[curLayer] - curAdversarialPredsHLRGD[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsHLRGD[curLayer+1]), axis=(1,2,3))), axis=0)
        undefMAE = np.mean(K.eval(cosine_loss(curAdversarialPredsUndef[curLayer],curAdversarialPredsUndef[curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPredsUndef[curLayer] - curAdversarialPredsUndef[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsUndef[curLayer+1]), axis=(1,2,3))), axis=0)
        fimMAE = np.mean(K.eval(cosine_loss(curAdversarialPredsFIM[curLayer],curAdversarialPredsFIM[curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPredsFIM[curLayer] - curAdversarialPredsFIM[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsFIM[curLayer+1]), axis=(1,2,3))), axis=0)
        layerWiseAGNLDRMAEs.append(LDRMAE)
        layerWiseAGNKLMAEs.append(KLMAE)
        layerWiseAGNHLRGDMAEs.append(hlrgdMAE)
        layerWiseAGNUndefMAEs.append(undefMAE)
        layerWiseAGNFIMMAEs.append(fimMAE)
        if (noisyTraining):
            awgnMAE = np.mean(K.eval(cosine_loss(curAdversarialPredsAGN [curLayer],curAdversarialPredsAGN [curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPredsAGN[curLayer] - curAdversarialPredsAGN[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsAGN[curLayer+1]), axis=(1,2,3))), axis=0)
            layerWiseAGNAGNMAEs.append(awgnMAE)

    #plot AGN perturbations
    plt.figure()
    numberOfLayers = len(layerWiseAGNUndefMAEs)
    curLevel =curNoisePower
    curCVIt = cvIndex
    plt.plot(range(numberOfLayers), layerWiseAGNUndefMAEs, marker='o', linestyle='-.', label=r'undefended $\rho$=%s p %s'%(str(curLevel*255./2.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseAGNLDRMAEs, marker='*', linestyle='-', label=r'LDR $\rho$=%s p %s'%(str(curLevel*255./2.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseAGNKLMAEs, marker='*', linestyle='-', label=r'DDSR $\rho$=%s p %s'%(str(curLevel*255./2.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseAGNHLRGDMAEs, marker='x', label=r'hlrgd $\rho$=%s p %s' %(str(curLevel*255./2.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseAGNFIMMAEs, marker='d', label=r'fim $\rho$=%s p %s' %(str(curLevel*255./2.), str(curCVIt)))
    if (noisyTraining):
        plt.plot(range(numberOfLayers), layerWiseAGNAGNMAEs, marker='v', label=r'awgn $\rho$=%s p %s' %(str(curLevel*255./2.), str(curCVIt)))
    plt.ylabel(r'layer-wise perturbation: mean of ($\frac{\vert h_i(x)-h_i(x_a)\vert}{\vert h_i(x) \vert} $)')
    plt.xlabel('hidden layer index ($i$)')
    plt.legend()
    plt.savefig('layerwiseAGNPerturbations%s_.png'%str(cvIndex))
    plt.close()
    # #plt.show()

    #fgsm induced perturbations
    #if we need to, recalculate the attacks
    if (advPowers[-1] != adversarialMAEBudget):
        testingAttacks = inputTestX + adversarialMAEBudget*fgsmData['test']
        testingAttacksResonatorsOff = inputTestX + adversarialMAEBudget*fgsmData['testResonatorsOff']
    #calculate adversarial perturbations (only for the largest budget, which is currently curNoisePower
    curAdversarialPreds = l1ProtectedModel.hiddenModel.predict([testX[:numberOfPertSamples, :, :, :], testingAttacksResonatorsOff[:numberOfPertSamples, :, :, :] if explicitDataDependence else testingAttacks[:numberOfPertSamples, :, :, :]])
    curAdversarialPredsDDSR = ddsrProtectedModel.hiddenModel.predict([testX[:numberOfPertSamples, :, :, :], testingAttacks[:numberOfPertSamples, :, :, :]])
    curAdversarialPredsHLRGD = hlrgd.ourVGG.hiddenModel.predict([testX[:numberOfPertSamples, :, :, :], testingAttacksResonatorsOff[:numberOfPertSamples, :, :, :] if explicitDataDependence else testingAttacks[:numberOfPertSamples, :, :, :]])
    curAdversarialPredsUndef = naiveModel.hiddenModel.predict([testX[:numberOfPertSamples, :, :, :], testingAttacksResonatorsOff[:numberOfPertSamples, :, :, :] if explicitDataDependence else testingAttacks[:numberOfPertSamples, :, :, :]])
    curAdversarialPredsFIM = eigenRegModel.hiddenModel.predict([testX[:numberOfPertSamples, :, :, :], testingAttacksResonatorsOff[:numberOfPertSamples, :, :, :] if explicitDataDependence else testingAttacks[:numberOfPertSamples, :, :, :]])
    if (noisyTraining):
        curAdversarialPredsAGN = awgnModel.hiddenModel.predict([testX[:numberOfPertSamples, :, :, :], testingAttacksResonatorsOff[:numberOfPertSamples, :, :, :] if explicitDataDependence else testingAttacks[:numberOfPertSamples, :, :, :]])
    layerWiseLDRMAEs, layerWiseKLMAEs, layerWiseHLRGDMAEs, layerWiseUndefMAEs, layerWiseFIMMAEs, layerWiseAGNMAEs = [], [], [], [], [], []
    for curLayer in list(range(len(curAdversarialPreds)))[1::2]:
        curAdversarialMAEs = []

        LDRMAE = np.mean(K.eval(cosine_loss(curAdversarialPreds[curLayer],curAdversarialPreds[curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPreds[curLayer] - curAdversarialPreds[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPreds[curLayer+1]), axis=(1,2,3))), axis=0)
        KLMAE = np.mean(K.eval(cosine_loss(curAdversarialPredsDDSR[curLayer],curAdversarialPredsDDSR[curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPredsDDSR[curLayer] - curAdversarialPredsDDSR[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsDDSR[curLayer+1]), axis=(1,2,3))), axis=0)
        hlrgdMAE = np.mean(K.eval(cosine_loss(curAdversarialPredsHLRGD[curLayer],curAdversarialPredsHLRGD[curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPredsHLRGD[curLayer] - curAdversarialPredsHLRGD[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsHLRGD[curLayer+1]), axis=(1,2,3))), axis=0)
        undefMAE = np.mean(K.eval(cosine_loss(curAdversarialPredsUndef[curLayer],curAdversarialPredsUndef[curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPredsUndef[curLayer] - curAdversarialPredsUndef[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsUndef[curLayer+1]), axis=(1,2,3))), axis=0)
        fimMAE = np.mean(K.eval(cosine_loss(curAdversarialPredsFIM[curLayer],curAdversarialPredsFIM[curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPredsFIM[curLayer] - curAdversarialPredsFIM[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsFIM[curLayer+1]), axis=(1,2,3))), axis=0)

        layerWiseLDRMAEs.append(LDRMAE)
        layerWiseKLMAEs.append(KLMAE)
        layerWiseHLRGDMAEs.append(hlrgdMAE)
        layerWiseUndefMAEs.append(undefMAE)
        layerWiseFIMMAEs.append(fimMAE)
        if (noisyTraining):
            layerWiseAGNMAE = np.mean(K.eval(cosine_loss(curAdversarialPredsAGN[curLayer],curAdversarialPredsAGN[curLayer + 1])))#np.mean((np.sum(np.abs(curAdversarialPredsAGN[curLayer] - curAdversarialPredsAGN[curLayer + 1]), axis=(1,2,3))/np.sum(np.abs(curAdversarialPredsAGN[curLayer+1]), axis=(1,2,3))), axis=0)
            layerWiseAGNMAEs.append(layerWiseAGNMAE)
    
    #free memory consumed by calculating the adversarial and agn perturbations

    #plot adversarial perturbations
    plt.figure()
    curLevel =curNoisePower
    curCVIt = cvIndex
    plt.plot(range(numberOfLayers), layerWiseUndefMAEs, marker='o', linestyle='-.', label=r'undefended $\epsilon$=%s p %s'%(str(curLevel*255./2.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseLDRMAEs, marker='*', linestyle='-', label=r'LDR $\epsilon$=%s p %s'%(str(curLevel*255./2.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseKLMAEs, marker='*', linestyle='-', label=r'DDSR $\epsilon$=%s p %s'%(str(curLevel*255./2.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseHLRGDMAEs, marker='x', label=r'hlrgd $\epsilon$=%s p %s' %(str(curLevel*255./2.), str(curCVIt)))
    plt.plot(range(numberOfLayers), layerWiseFIMMAEs, marker='d', label=r'fim $\epsilon$=%s p %s' %(str(curLevel*255./2.), str(curCVIt)))
    if (noisyTraining):
        plt.plot(range(numberOfLayers), layerWiseAGNMAEs, marker='v', label=r'fim $\epsilon$=%s p %s' %(str(curLevel*255./2.), str(curCVIt)))
    plt.ylabel(r'layer-wise perturbation: mean of ($\frac{\vert h_i(x)-h_i(x_a)\vert}{\vert h_i(x) \vert} $)')
    plt.xlabel('hidden layer index ($i$)')
    plt.legend()
    plt.savefig('layerwisePerturbations%s_.png'%str(cvIndex))
    plt.close()
    #plt.show()


    crossValResults['Acc'][cvIndex] = allAdvAccuracies
    crossValResults['noiseAcc'][cvIndex] = allNoiseAccuracies

    crossValResults['averageLayerwiseUndefMAEs'][cvIndex] = layerWiseUndefMAEs
    crossValResults['averageLayerwiseLDRMAEs'][cvIndex] = layerWiseLDRMAEs
    crossValResults['averageLayerwiseKLMAEs'][cvIndex] = layerWiseKLMAEs
    crossValResults['averageLayerwiseHLRGDMAEs'][cvIndex] = layerWiseHLRGDMAEs
    crossValResults['averageLayerwiseFIMMAEs'][cvIndex] = layerWiseFIMMAEs
    crossValResults['averageLayerwiseAGNUndefMAEs'][cvIndex] = layerWiseAGNUndefMAEs
    crossValResults['averageLayerwiseAGNLDRMAEs'][cvIndex] = layerWiseAGNLDRMAEs
    crossValResults['averageLayerwiseAGNKLMAEs'][cvIndex] = layerWiseAGNKLMAEs
    crossValResults['averageLayerwiseAGNHLRGDMAEs'][cvIndex] = layerWiseAGNHLRGDMAEs
    crossValResults['averageLayerwiseAGNFIMMAEs'][cvIndex] = layerWiseAGNFIMMAEs
    if (noisyTraining):
        crossValResults['averageLayerwiseAGNMAEs'][cvIndex] = layerWiseAGNMAEs
        crossValResults['averageLayerwiseAGNAGNMAEs'][cvIndex] = layerWiseAGNAGNMAEs

    #pickle the results
    pickle.dump(crossValResults, open('crossValResults.pickle', 'wb'))

    #convert acc results into numpy arrays for easy arithmetic
    advAccuracy['undefended'] = np.array(advAccuracy['undefended'])
    advAccuracy['LDR'] = np.array(advAccuracy['LDR'])
    if multipleForwardPasses is not None:
        advAccuracy['ddsr'] = np.array(advAccuracy['ddsr_singleFP'])
        noiseAccuracy['ddsr'] = np.array(noiseAccuracy['ddsr_singleFP'])
        advAccuracy['ddsr_MFP'] = np.array(advAccuracy['ddsr_MFP'])
        noiseAccuracy['ddsr_MFP'] = np.array(noiseAccuracy['ddsr_MFP'])
    else:
        advAccuracy['ddsr'] = np.array(advAccuracy['ddsr'])
        noiseAccuracy['ddsr'] = np.array(noiseAccuracy['ddsr'])

    advAccuracy['hlrgd'] = np.array(advAccuracy['hlrgd'])
    advAccuracy['fim'] = np.array(advAccuracy['fim'])
    noiseAccuracy['undefended'] = np.array(noiseAccuracy['undefended'])
    noiseAccuracy['LDR'] = np.array(noiseAccuracy['LDR'])

    noiseAccuracy['hlrgd'] = np.array(noiseAccuracy['hlrgd'])
    noiseAccuracy['fim'] = np.array(noiseAccuracy['fim'])
    if (noisyTraining):
        noiseAccuracy['agn'] = np.array(noiseAccuracy['agn'])
        advAccuracy['agn'] = np.array(advAccuracy['agn'])


    
    #plot noise acc results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('AGN accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('noise power (p)')
    noisePowersPlot = copy.copy(noisePowers)
    noisePowersPlot.insert(0, 0)
    # advPowersAx = np.array(np.sqrt(advPowersPlot))*(255./2.)
    plt.plot(noisePowersPlot, noiseAccuracy['LDR'][-1], label='LDR')
    plt.plot(noisePowersPlot, noiseAccuracy['ddsr'][-1], label='DDSR')
    plt.plot(noisePowersPlot, noiseAccuracy['hlrgd'][-1], label='HLRGD')
    plt.plot(noisePowersPlot, noiseAccuracy['fim'][-1], label='FIM')
    if (noisyTraining):
        plt.plot(noisePowersPlot, noiseAccuracy['agn'][-1], label='awgn')
    plt.legend()
    plt.savefig('acc_vs_budget_cv_%sAGN.png' % str(cvIndex))
    texts = []
    for i in range(noiseAccuracy['undefended'].shape[0]):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['undefended'][i]))))
    for i in range(noiseAccuracy['LDR'].shape[1]):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['LDR'][-1][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['LDR'][-1][i]))))
    for i in range(noiseAccuracy['ddsr'].shape[1]):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['ddsr'][-1][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['ddsr'][-1][i]))))
    for i in range(noiseAccuracy['hlrgd'].shape[1]):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['hlrgd'][-1][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['hlrgd'][-1][i]))))
    for i in range(noiseAccuracy['fim'].shape[1]):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['fim'][-1][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['fim'][-1][i]))))
    if (noisyTraining):
        for i in range(noiseAccuracy['agn'].shape[1]):
                texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['agn'][-1][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['agn'][-1][i]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('acc_vs_budget_cv_%sAGN.png'%str(cvIndex))
    plt.close()
    #plt.show()


    #noise marginal improvement plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('AGN improvement')
    plt.ylabel('marginal improvement in accuracy')
    plt.xlabel('noise power (p)')
    plt.plot(noisePowersPlot, noiseAccuracy['LDR'][-1]-noiseAccuracy['undefended'][:], label='LDR')
    plt.plot(noisePowersPlot, noiseAccuracy['ddsr'][-1]-noiseAccuracy['undefended'][:], label='DDSR')
    plt.plot(noisePowersPlot, noiseAccuracy['hlrgd'][-1]-noiseAccuracy['undefended'][:], label='HLRGD')
    plt.plot(noisePowersPlot, noiseAccuracy['fim'][-1]-noiseAccuracy['undefended'][:], label='FIM')
    if (noisyTraining):
        plt.plot(noisePowersPlot, noiseAccuracy['agn'][-1]-noiseAccuracy['undefended'][:], label='AGN')
    plt.legend()
    plt.savefig('marg_improv_vs_budget_%sAGN.png' % str(cvIndex))
    texts = []
    # for i in range(len(advAccuracy['undefended'])):
    #     texts.append(plt.text(advPowersAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowers[i]), str(advAccuracy['undefended'][i]))))
    for i in range(noiseAccuracy['LDR'].shape[1]):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['LDR'][-1][i]-noiseAccuracy['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['LDR'][-1][i]))))
    for i in range(noiseAccuracy['ddsr'].shape[1]):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['ddsr'][-1][i]-noiseAccuracy['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['ddsr'][-1][i]))))
    for i in range(noiseAccuracy['hlrgd'].shape[1]):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['hlrgd'][-1][i]-noiseAccuracy['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['hlrgd'][-1][i]))))
    for i in range(noiseAccuracy['fim'].shape[1]):
        texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['fim'][-1][i]-noiseAccuracy['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['fim'][-1][i]))))
    if (noisyTraining):
        for i in range(noiseAccuracy['agn'].shape[1]):
            texts.append(plt.text(noisePowersPlot[i], noiseAccuracy['agn'][-1][i]-noiseAccuracy['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(noiseAccuracy['agn'][-1][i]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('marg_improv_vs_budget_%sAGN_withLabels.png'%str(cvIndex))
    plt.close()
    #plt.show()


    #plot acc results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('adversarial accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('adversarial budget')
    advPowersPlot = copy.copy(advPowers)
    advPowersPlot.insert(0, 0)
    advPowersAx = np.array(advPowersPlot)*(255./2.)
    plt.plot(advPowersAx, advAccuracy['LDR'][-1], label='LDR')
    plt.plot(advPowersAx, advAccuracy['ddsr'][-1], label='DDSR')
    if multipleForwardPasses is not None:
        plt.plot(advPowersAx, advAccuracy['ddsr_MFP'][-1], label='DDSR %s passes'%str(multipleForwardPasses))
    plt.plot(advPowersAx, advAccuracy['hlrgd'][-1], label='HLRGD')
    plt.plot(advPowersAx, advAccuracy['fim'][-1], label='FIM')
    if (noisyTraining):
        plt.plot(advPowersAx, advAccuracy['agn'][-1], label='AGN')
    plt.legend()
    plt.savefig('acc_vs_budget_cv_%s.png' % str(cvIndex))
    texts = []
    for i in range(advAccuracy['undefended'].shape[0]):
        texts.append(plt.text(advPowersAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(advAccuracy['undefended'][i]))))
    for i in range(advAccuracy['LDR'].shape[1]):
        texts.append(plt.text(advPowersAx[i], advAccuracy['LDR'][-1][i], '(%s, %s)'%(str(advPowersPlot[i]), str(advAccuracy['LDR'][-1][i]))))
    for i in range(advAccuracy['ddsr'].shape[1]):
        texts.append(plt.text(advPowersAx[i], advAccuracy['ddsr'][-1][i], '(%s, %s)'%(str(advPowersPlot[i]), str(advAccuracy['ddsr'][-1][i]))))
    for i in range(advAccuracy['hlrgd'].shape[1]):
        texts.append(plt.text(advPowersAx[i], advAccuracy['hlrgd'][-1][i], '(%s, %s)'%(str(advPowersPlot[i]), str(advAccuracy['hlrgd'][-1][i]))))
    for i in range(advAccuracy['fim'].shape[1]):
        texts.append(plt.text(advPowersAx[i], advAccuracy['fim'][-1][i], '(%s, %s)'%(str(advPowersPlot[i]), str(advAccuracy['fim'][-1][i]))))
    if (noisyTraining):
        for i in range(advAccuracy['agn'].shape[1]):
            texts.append(plt.text(advPowersAx[i], advAccuracy['agn'][-1][i], '(%s, %s)'%(str(advPowersPlot[i]), str(advAccuracy['agn'][-1][i]))))
    plt.legend()
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('acc_vs_budget_cv_%s_withLabels.png'%str(cvIndex))
    plt.close()
    #plt.show()


    #marginal improvement plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('adversarial accuracy')
    plt.ylabel('marginal improvement in accuracy')
    plt.xlabel('adversarial budget (p)')
    plt.plot(np.array(advPowersPlot)*(255./2.), advAccuracy['LDR'][-1]-advAccuracy['undefended'], label='LDR')
    plt.plot(np.array(advPowersPlot)*(255./2.), advAccuracy['ddsr'][-1]-advAccuracy['undefended'], label='DDSR')
    plt.plot(np.array(advPowersPlot)*(255./2.), advAccuracy['hlrgd'][-1]-advAccuracy['undefended'], label='HLRGD')
    plt.plot(np.array(advPowersPlot) * (255. / 2.), advAccuracy['fim'][-1]- advAccuracy['undefended'], label='FIM')
    if (noisyTraining):
        plt.plot(np.array(advPowersPlot) * (255. / 2.), advAccuracy['agn'][-1]- advAccuracy['undefended'], label='AGN')
    plt.legend()
    plt.savefig('marg_improv_vs_budget_%s.png' % str(cvIndex))
    texts = []
    # for i in range(len(advAccuracy['undefended'])):
    #     texts.append(plt.text(advPowersAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowers[i]), str(advAccuracy['undefended'][i]))))
    for i in range(advAccuracy['LDR'].shape[1]):
        texts.append(plt.text(advPowersAx[i], advAccuracy['LDR'][-1][i]-advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(advAccuracy['LDR'][-1][i]))))
    for i in range(advAccuracy['ddsr'].shape[1]):
        texts.append(plt.text(advPowersAx[i], advAccuracy['ddsr'][-1][i]-advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(advAccuracy['ddsr'][-1][i]))))
    for i in range(advAccuracy['hlrgd'].shape[1]):
        texts.append(plt.text(advPowersAx[i], advAccuracy['hlrgd'][-1][i]-advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(advAccuracy['hlrgd'][-1][i]))))
    for i in range(advAccuracy['fim'].shape[1]):
        texts.append(plt.text(advPowersAx[i], advAccuracy['fim'][-1][i]-advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(advAccuracy['fim'][-1][i]))))
    if (noisyTraining):
        for i in range(advAccuracy['agn'].shape[1]):
            texts.append(plt.text(advPowersAx[i], advAccuracy['agn'][-1][i]-advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(advAccuracy['agn'][-1][i]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('marg_improv_vs_budget_%s_withLabels.png'%str(cvIndex))
    plt.close()
    #plt.show()


    #fix garbage collection
    # if (cvIndex < numberOfReplications - 1):
    #     naiveModel.deleteModels()
    #     del naiveModel
    #     for curKey in list(fgsmData.keys()):
    #         del fgsmData[curKey]
    #     #tf.compat.v1.reset_default_graph()
    #tf.keras.backend.clear_session()
    #     print(gc.collect())         
    #delete old models and clear the backend so we don't fill up the gpu memory
    if (cvIndex < numberOfReplications-1):
        if (noisyTraining):
            awgnModel.deleteModels()
            del awgnModel
        naiveModel.deleteModels()
        hlrgd.deleteModels()
        ddsrProtectedModel.deleteModels()
        l1ProtectedModel.deleteModels()
        del naiveModel
        del hlrgd
        del ddsrProtectedModel
        del l1ProtectedModel
        del trainX
        del testX
        for curKey in list(fgsmData.keys()):
            del fgsmData[curKey]
        #tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        print(gc.collect())         

    cvIndex += 1
