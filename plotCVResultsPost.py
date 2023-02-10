########################################################################################################################
########################################################################################################################
# Authors: David Schwartz
# This script plots the results of a CV experimen   t after the experiment is complete
#
#
#
########################################################################################################################
########################################################################################################################

########################################################################################################################
#imports
import os, sys, copy
import pickle

import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
########################################################################################################################


########################################################################################################################
#use this for now but fix it once the next round is done
# advPowers = [0.01, 0.025, 0.05, 0.1, 0.25]
#noisePowers = list(np.arange(0.05, 0.35, 0.1))
#load data from the relevant pickle file
cvResultsFileName = 'fMNcrossValResults'#'unregCifar10'#'fMNcrossValResults'#'unregCifar10'#'crossValResults'#'ddsr_paper_crossValResults_ddsr_model_also_regularized_HLDR_cifar10'#'fMNcrossValResults'#'ddsr_paper_crossValResults_ddsr_model_also_regularized_HLDR_cifar10'#'crossValResults'#'5_17_paper'#'crossValResults'#'ddsr_explicit_crossValResults'

pickleFile = pickle.load(open(os.path.join(os.getcwd(),cvResultsFileName+'.pickle'), 'rb'))
crossValResults = copy.copy(pickleFile)
noisyTraining = crossValResults['noisyTraining']
modelList = ['hldr', 'fim', 'hlrgd']
if noisyTraining:
    modelList.append('agn')

keyList = list(crossValResults.keys())
print("keys")
print(keyList)
# print(list(crossValResults['averageLayerwiseUndefMAEs'].keys()))
# print(crossValResults['averageLayerwiseUndefMAEs'][0])
print("data loaded")
advPowers = crossValResults['budgets'] 
noisePowers = crossValResults['noiseBudgets']
if 'testPerfVsEpochs' in keyList:
    testPerfVsEpochs = crossValResults['testPerfVsEpochs']
else:
    testPerfVsEpochs = True

if 'testBetweenEpochs' in keyList:
    testBetweenEpochs = crossValResults['testBetweenEpochs']
else:
    testBetweenEpochs = 1
########################################################################################################################

########################################################################################################################
#load variables
numberOfFolds = len(list(crossValResults['Acc'].keys()))
maeCVKey = list(crossValResults['averageLayerwiseUndefMAEs'].keys())[0]
numberOfLayers = len(crossValResults['averageLayerwiseUndefMAEs'][maeCVKey])
aggregatedResultsMats = dict()

argsDict = crossValResults['argsDict']
#load variables
numberOfFolds = len(list(crossValResults['Acc'].keys()))
aggregatedResultsMats = dict()

argsDict = crossValResults['argsDict']
print('*****************************************************************************************************************')
print("arguments")
print("")
for curKey in list(argsDict.keys()):
    print("key: %s"%str(curKey))
    print("arg: %s"%str(argsDict[curKey]))
    
print('*****************************************************************************************************************')
chosenReg = argsDict['chosenReg']
explicitDataDependence = argsDict['explicitDataDependence'] if 'explicitDataDependence' in list(argsDict.keys()) else True
if chosenReg is None:
    if explicitDataDependence:
        chosenReg = 'DDSR'
    else:
        choseenReg = 'IDDSR'
#if we only test performance after all epochs are complete
if (not testPerfVsEpochs):
    numberOfNoiseMagnitudes = len(crossValResults['Acc'][0]['undefended'])
    numberOfAGNMagnitudes = len(crossValResults['noiseAcc'][0]['undefended'])
    #collect aggregated results across CV experiment
    aggregatedAccsUndef, aggregatedAccsFIM, aggregatedAccsHLDR, aggregatedAccsKLDR, aggregatedAccsHLRGD, aggregatedAccsAGN = -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes))
    aggregatedMAEsUndef, aggregatedMAEsFIM, aggregatedMAEsHLDR, aggregatedMAEsKLDR, aggregatedMAEsHLRGD, aggregatedMAEsAGN = -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfLayers)), \
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfLayers))
    aggregatedAGNAccsUndef, aggregatedAGNAccsFIM, aggregatedAGNAccsHLDR, aggregatedAGNAccsKLDR, aggregatedAGNAccsHLRGD, aggregatedAGNAccsAGN = -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes)), \
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes)),\
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes)),\
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes)), \
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes)),\
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes))
    aggregatedAGNMAEsUndef, aggregatedAGNMAEsFIM, aggregatedAGNMAEsHLDR, aggregatedAGNMAEsKLDR, aggregatedAGNMAEsHLRGD, aggregatedAGNMAEsAGN = -1*np.ones(shape=(numberOfFolds, numberOfLayers)), \
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfLayers)), \
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfLayers))
    for cvIndex in range(numberOfFolds):
        #aggregate adversarial results
        aggregatedAccsUndef[cvIndex, :] = crossValResults['Acc'][cvIndex]['undefended']
        aggregatedAccsFIM[cvIndex, :] = crossValResults['Acc'][cvIndex]['fim']
        aggregatedAccsHLDR[cvIndex, :] = crossValResults['Acc'][cvIndex]['hldr']
        aggregatedAccsKLDR[cvIndex, :] = crossValResults['Acc'][cvIndex]['ddsr']
        aggregatedAccsHLRGD[cvIndex, :] = crossValResults['Acc'][cvIndex]['hlrgd']
        aggregatedAccsAGN[cvIndex, :] = crossValResults['Acc'][cvIndex]['agn']
        aggregatedMAEsUndef[cvIndex, :] = np.array(crossValResults['averageLayerwiseUndefMAEs'][cvIndex])
        aggregatedMAEsFIM[cvIndex, :] = np.array(crossValResults['averageLayerwiseFIMMAEs'][cvIndex])
        aggregatedMAEsHLDR[cvIndex, :] = np.array(crossValResults['averageLayerwiseHLDRMAEs'][cvIndex])
        aggregatedMAEsKLDR[cvIndex, :] = np.array(crossValResults['averageLayerwiseKLMAEs'][cvIndex])
        aggregatedMAEsHLRGD[cvIndex, :] = np.array(crossValResults['averageLayerwiseHLRGDMAEs'][cvIndex])
        if (noisyTraining):
            aggregatedMAEsAGN[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNMAEs'][cvIndex])
        #aggregate AGN results
        aggregatedAGNAccsUndef[cvIndex, :] = crossValResults['noiseAcc'][cvIndex]['undefended']
        aggregatedAGNAccsFIM[cvIndex, :] = crossValResults['noiseAcc'][cvIndex]['fim']
        aggregatedAGNAccsHLDR[cvIndex, :] = crossValResults['noiseAcc'][cvIndex]['hldr']
        aggregatedAGNAccsKLDR[cvIndex, :] = crossValResults['noiseAcc'][cvIndex]['ddsr']
        aggregatedAGNAccsHLRGD[cvIndex, :] = crossValResults['noiseAcc'][cvIndex]['hlrgd']
        aggregatedAGNAccsAGN[cvIndex, :] = crossValResults['noiseAcc'][cvIndex]['agn']
        aggregatedAGNMAEsUndef[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNUndefMAEs'][cvIndex])
        aggregatedAGNMAEsFIM[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNFIMMAEs'][cvIndex])
        aggregatedAGNMAEsHLDR[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNHLDRMAEs'][cvIndex])
        aggregatedAGNMAEsKLDR[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNKLMAEs'][cvIndex])
        aggregatedAGNMAEsHLRGD[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNHLRGDMAEs'][cvIndex])
        if (noisyTraining):
            aggregatedAGNMAEsAGN[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNAGNMAEs'][cvIndex])

#experimental data extraction is different if we test the model between training epochs
elif (testPerfVsEpochs):
    numberOfNoiseMagnitudes = crossValResults['Acc'][0]['undefended'].shape[0]
    numberOfAGNMagnitudes = crossValResults['noiseAcc'][0]['undefended'].shape[0]

    allAdvAccuracies = crossValResults['Acc']
    allNoiseAccuracies = crossValResults['noiseAcc']
    numberOfEpochs = crossValResults['Acc'][0]['hldr'].shape[0]
    # allAdvAccuracies['undefended']
    # allAdvAccuracies['hldr'] = [np.array(advAccuracy['hldr'])]
    # allAdvAccuracies['hlrgd'] = [np.array(advAccuracy['hlrgd'])]
    # allAdvAccuracies['fim'] = [np.array(advAccuracy['fim'])]
    # allNoiseAccuracies['undefended'] = [np.array(noiseAccuracy['undefended'])]
    # allNoiseAccuracies['hldr'] = [np.array(noiseAccuracy['hldr'])]
    # allNoiseAccuracies['hlrgd'] = [np.array(noiseAccuracy['hlrgd'])]
    # allNoiseAccuracies['fim']

    #collect aggregated results across CV experiment    
    aggregatedAccsUndef, aggregatedAccsFIM, aggregatedAccsHLDR,aggregatedAccsKLDR, aggregatedAccsHLRGD, aggregatedAccsAGN = -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes, numberOfEpochs)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes, numberOfEpochs)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes, numberOfEpochs)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes, numberOfEpochs)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes, numberOfEpochs)), \
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfNoiseMagnitudes, numberOfEpochs))
    aggregatedMAEsUndef, aggregatedMAEsFIM, aggregatedMAEsHLDR,aggregatedMAEsKLDR, aggregatedMAEsHLRGD, aggregatedMAEsAGN = -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfLayers)), \
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfLayers)), \
                                                                                                        -1*np.ones(shape=(numberOfFolds, numberOfLayers))
    aggregatedAGNAccsUndef, aggregatedAGNAccsFIM, aggregatedAGNAccsHLDR,aggregatedAGNAccsKLDR, aggregatedAGNAccsHLRGD, aggregatedAGNAccsAGN = -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes, numberOfEpochs)), \
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes, numberOfEpochs)),\
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes, numberOfEpochs)),\
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes, numberOfEpochs)), \
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes, numberOfEpochs)), \
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfAGNMagnitudes, numberOfEpochs))
    aggregatedAGNMAEsUndef, aggregatedAGNMAEsFIM, aggregatedAGNMAEsHLDR,aggregatedAGNMAEsKLDR, aggregatedAGNMAEsHLRGD, aggregatedAGNMAEsAGN = -1*np.ones(shape=(numberOfFolds, numberOfLayers)), \
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfLayers)),\
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfLayers)), \
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfLayers)), \
                                                                                                                            -1*np.ones(shape=(numberOfFolds, numberOfLayers))

    for cvIndex in range(numberOfFolds):
        aggregatedMAEsUndef[cvIndex, :] = np.array(crossValResults['averageLayerwiseUndefMAEs'][cvIndex])
        aggregatedMAEsFIM[cvIndex, :] = np.array(crossValResults['averageLayerwiseFIMMAEs'][cvIndex])
        aggregatedMAEsHLDR[cvIndex, :] = np.array(crossValResults['averageLayerwiseHLDRMAEs'][cvIndex])
        aggregatedMAEsKLDR[cvIndex, :] = np.array(crossValResults['averageLayerwiseKLMAEs'][cvIndex])
        aggregatedMAEsHLRGD[cvIndex, :] = np.array(crossValResults['averageLayerwiseHLRGDMAEs'][cvIndex])
        aggregatedAGNMAEsUndef[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNUndefMAEs'][cvIndex])
        aggregatedAGNMAEsFIM[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNFIMMAEs'][cvIndex])
        aggregatedAGNMAEsHLDR[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNHLDRMAEs'][cvIndex])
        aggregatedAGNMAEsKLDR[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNKLMAEs'][cvIndex])
        aggregatedAGNMAEsHLRGD[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNHLRGDMAEs'][cvIndex])
        if (noisyTraining):
            aggregatedAGNMAEsAGN[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNAGNMAEs'][cvIndex])
        if (noisyTraining):
            aggregatedMAEsAGN[cvIndex, :] = np.array(crossValResults['averageLayerwiseAGNMAEs'][cvIndex])

        for curEpoch in range(numberOfEpochs):
            print(crossValResults['Acc'][cvIndex]['undefended'].shape)
            #aggregate adversarial results
            aggregatedAccsUndef[cvIndex, :, curEpoch] = crossValResults['Acc'][cvIndex]['undefended']
            aggregatedAccsFIM[cvIndex, :, curEpoch] = crossValResults['Acc'][cvIndex]['fim'][curEpoch if (crossValResults['Acc'][cvIndex]['fim'].shape[0] > curEpoch) else crossValResults['Acc'][cvIndex]['fim'].shape[0]-1,:]
            aggregatedAccsHLDR[cvIndex, :, curEpoch] = crossValResults['Acc'][cvIndex]['hldr'][curEpoch, :]
            aggregatedAccsKLDR[cvIndex, :, curEpoch] = crossValResults['Acc'][cvIndex]['ddsr'][curEpoch, :]
            aggregatedAccsHLRGD[cvIndex, :, curEpoch] = crossValResults['Acc'][cvIndex]['hlrgd'][curEpoch, :]
            aggregatedAccsAGN[cvIndex, :, curEpoch] = crossValResults['Acc'][cvIndex]['agn'][curEpoch, :]

            #aggregate AGN results
            aggregatedAGNAccsUndef[cvIndex, :, curEpoch] = crossValResults['noiseAcc'][cvIndex]['undefended']
            aggregatedAGNAccsFIM[cvIndex, :, curEpoch] = crossValResults['noiseAcc'][cvIndex]['fim'][curEpoch if (crossValResults['Acc'][cvIndex]['fim'].shape[0] > curEpoch) else crossValResults['Acc'][cvIndex]['fim'].shape[0]-1, :]
            aggregatedAGNAccsHLDR[cvIndex, :, curEpoch] = crossValResults['noiseAcc'][cvIndex]['hldr'][curEpoch, :]
            aggregatedAGNAccsKLDR[cvIndex, :, curEpoch] = crossValResults['noiseAcc'][cvIndex]['ddsr'][curEpoch, :]
            aggregatedAGNAccsHLRGD[cvIndex, :, curEpoch] = crossValResults['noiseAcc'][cvIndex]['hlrgd'][curEpoch, :]
            aggregatedAGNAccsAGN[cvIndex, :, curEpoch] = crossValResults['noiseAcc'][cvIndex]['agn'][curEpoch, :]

    
#calculate aggregated statistics of interest (i.e. mean across the cvIndex axis)
meanAdvAccs, stdAdvAccs = dict(), dict()
meanNoiseAccs, stdNoiseAccs = dict(), dict()

meanAdvAccs['undefended'] = np.mean(aggregatedAccsUndef, axis=0)
meanAdvAccs['fim'] = np.mean(aggregatedAccsFIM, axis=0)
meanAdvAccs['hldr'] = np.mean(aggregatedAccsHLDR, axis=0)
meanAdvAccs['kldr'] = np.mean(aggregatedAccsKLDR, axis=0)
meanAdvAccs['hlrgd'] = np.mean(aggregatedAccsHLRGD, axis=0)
if (noisyTraining):
    meanAdvAccs['agn'] = np.mean(aggregatedAccsAGN, axis=0)

meanNoiseAccs['undefended'] = np.mean(aggregatedAGNAccsUndef, axis=0)
meanNoiseAccs['fim'] = np.mean(aggregatedAGNAccsFIM, axis=0)
meanNoiseAccs['hldr'] = np.mean(aggregatedAGNAccsHLDR, axis=0)
meanNoiseAccs['kldr'] = np.mean(aggregatedAGNAccsKLDR, axis=0)
meanNoiseAccs['hlrgd'] = np.mean(aggregatedAGNAccsHLRGD, axis=0)
if (noisyTraining):
    meanNoiseAccs['agn'] = np.mean(aggregatedAGNAccsAGN, axis=0)

stdAdvAccs['undefended'] = 1.96*np.std(aggregatedAccsUndef, axis=0)/np.sqrt(numberOfFolds)
stdAdvAccs['fim'] = 1.96*np.std(aggregatedAccsFIM, axis=0)/np.sqrt(numberOfFolds)
stdAdvAccs['hldr'] = 1.96*np.std(aggregatedAccsHLDR, axis=0)/np.sqrt(numberOfFolds)
stdAdvAccs['kldr'] = 1.96*np.std(aggregatedAccsKLDR, axis=0)/np.sqrt(numberOfFolds)
stdAdvAccs['hlrgd'] = 1.96*np.std(aggregatedAccsHLRGD, axis=0)/np.sqrt(numberOfFolds)
if (noisyTraining):
    stdAdvAccs['agn'] = 1.96*np.std(aggregatedAccsAGN, axis=0)/np.sqrt(numberOfFolds)

stdNoiseAccs['undefended'] = 1.96*np.std(aggregatedAGNAccsUndef, axis=0)/np.sqrt(numberOfFolds)
stdNoiseAccs['fim'] = 1.96*np.std(aggregatedAGNAccsFIM, axis=0)/np.sqrt(numberOfFolds)
stdNoiseAccs['hldr'] = 1.96*np.std(aggregatedAGNAccsHLDR, axis=0)/np.sqrt(numberOfFolds)
stdNoiseAccs['kldr'] = 1.96*np.std(aggregatedAGNAccsKLDR, axis=0)/np.sqrt(numberOfFolds)
stdNoiseAccs['hlrgd'] = 1.96*np.std(aggregatedAGNAccsHLRGD, axis=0)/np.sqrt(numberOfFolds)
if (noisyTraining):
    stdNoiseAccs['agn'] = 1.96*np.std(aggregatedAGNAccsAGN, axis=0)/np.sqrt(numberOfFolds)

meanAverageLayerwiseUndefMAEs = np.mean(aggregatedMAEsUndef, axis=0)
meanAverageLayerwiseHLDRMAEs = np.mean(aggregatedMAEsHLDR, axis=0)
meanAverageLayerwiseKLDRMAEs = np.mean(aggregatedMAEsKLDR, axis=0)
meanAverageLayerwiseHLRGDMAEs = np.mean(aggregatedMAEsHLRGD, axis=0)
meanAverageLayerwiseFIMMAEs = np.mean(aggregatedMAEsFIM, axis=0)
if (noisyTraining):
    meanAverageLayerwiseAGNMAEs = np.mean(aggregatedMAEsAGN, axis=0)

meanAverageLayerwiseAGNUndefMAEs = np.mean(aggregatedAGNMAEsUndef, axis=0)
meanAverageLayerwiseAGNHLDRMAEs = np.mean(aggregatedAGNMAEsHLDR, axis=0)
meanAverageLayerwiseAGNKLDRMAEs = np.mean(aggregatedAGNMAEsKLDR, axis=0)
meanAverageLayerwiseAGNHLRGDMAEs = np.mean(aggregatedAGNMAEsHLRGD, axis=0)
meanAverageLayerwiseAGNFIMMAEs = np.mean(aggregatedAGNMAEsFIM, axis=0)
if (noisyTraining):
    meanAverageLayerwiseAGNAGNMAEs = np.mean(aggregatedAGNMAEsAGN, axis=0)

stdAverageLayerwiseUndefMAEs = 1.96*np.std(aggregatedMAEsUndef, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseHLDRMAEs = 1.96*np.std(aggregatedMAEsHLDR, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseKLDRMAEs = 1.96*np.std(aggregatedMAEsKLDR, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseHLRGDMAEs = 1.96*np.std(aggregatedMAEsHLRGD, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseFIMMAEs = 1.96*np.std(aggregatedMAEsFIM, axis=0)/np.sqrt(numberOfFolds)
if (noisyTraining):
    stdAverageLayerwiseAGNMAEs = 1.96*np.std(aggregatedMAEsAGN, axis=0)/np.sqrt(numberOfFolds)

stdAverageLayerwiseAGNUndefMAEs = 1.96*np.std(aggregatedAGNMAEsUndef, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseAGNHLDRMAEs = 1.96*np.std(aggregatedAGNMAEsHLDR, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseAGNKLDRMAEs = 1.96*np.std(aggregatedAGNMAEsKLDR, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseAGNHLRGDMAEs = 1.96*np.std(aggregatedAGNMAEsHLRGD, axis=0)/np.sqrt(numberOfFolds)
stdAverageLayerwiseAGNFIMMAEs = 1.96*np.std(aggregatedAGNMAEsFIM, axis=0)/np.sqrt(numberOfFolds)
if (noisyTraining):
    stdAverageLayerwiseAGNAGNMAEs = 1.96*np.std(aggregatedAGNMAEsAGN, axis=0)/np.sqrt(numberOfFolds)
########################################################################################################################

########################################################################################################################
#if we only tested performance once after all training completed
if (not testPerfVsEpochs):
    #plot stuff
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.title('adversarial accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('adversarial budget ($\epsilon$)')
    advPowersPlot = copy.copy(advPowers)
    advPowersPlot.insert(0, 0)
    advPowersAx = np.array(advPowersPlot)
    plt.errorbar(advPowersAx, meanAdvAccs['hldr'], stdAdvAccs['hldr'], label='LDR', marker='1')
    plt.errorbar(advPowersAx, meanAdvAccs['kldr'], stdAdvAccs['kldr'], label=chosenReg, marker='2')
    plt.errorbar(advPowersAx, meanAdvAccs['hlrgd'], stdAdvAccs['hlrgd'], label='HGD', marker='3')
    plt.errorbar(advPowersAx, meanAdvAccs['fim'], stdAdvAccs['fim'], label='FIMR', marker='4')
    if (noisyTraining):
        plt.errorbar(advPowersAx, meanAdvAccs['agn'], stdAdvAccs['agn'], label='AGNT', marker='x')#linestyle='-.')
    plt.legend()
    plt.savefig('aggregatedACCVsBudget_no_labels.pdf')
    texts = []
    for i in range(len(meanAdvAccs['undefended'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['undefended'][i]))))
    for i in range(len(meanAdvAccs['hldr'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['hldr'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['hldr'][i]))))
    for i in range(len(meanAdvAccs['kldr'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['kldr'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['kldr'][i]))))
    for i in range(len(meanAdvAccs['hlrgd'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['hlrgd'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['hlrgd'][i]))))
    for i in range(len(meanAdvAccs['fim'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['fim'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['fim'][i]))))
    if (noisyTraining):
        for i in range(len(meanAdvAccs['agn'])):
            texts.append(plt.text(advPowersAx[i], meanAdvAccs['agn'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['agn'][i]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('aggregatedACCVsBudget_with_labels.pdf')
    #plt.draw()
    #plt.pause(0.01)


    #marginal improvement plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylabel('accuracy gain')
    plt.xlabel('adversarial budget ($\epsilon$)')
    plt.errorbar(np.array(advPowersPlot), meanAdvAccs['hldr']-meanAdvAccs['undefended'], stdAdvAccs['hldr'], label='LDR', marker='1')
    plt.errorbar(np.array(advPowersPlot), meanAdvAccs['kldr']-meanAdvAccs['undefended'], stdAdvAccs['kldr'], label=chosenReg, marker='2')
    plt.errorbar(np.array(advPowersPlot), meanAdvAccs['hlrgd']-meanAdvAccs['undefended'], stdAdvAccs['hlrgd'], label='HGD', marker='3')
    plt.errorbar(np.array(advPowersPlot), meanAdvAccs['fim'] - meanAdvAccs['undefended'], stdAdvAccs['fim'], label='FIMR', marker='4')
    if (noisyTraining):
        plt.errorbar(np.array(advPowersPlot), meanAdvAccs['agn'] - meanAdvAccs['undefended'], stdAdvAccs['agn'], label='AGNT', marker='x')
    plt.legend()
    plt.savefig('aggregated_marg_improv_vs_budget_no_labels.pdf')
    texts = []
    # for i in range(len(advAccuracy['undefended'])):
    #     texts.append(plt.text(advPowersAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowers[i]), str(advAccuracy['undefended'][i]))))
    for i in range(len(meanAdvAccs['hldr'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['hldr'][i]-meanAdvAccs['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['hldr'][i]))))
    for i in range(len(meanAdvAccs['kldr'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['kldr'][i]-meanAdvAccs['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['kldr'][i]))))
    for i in range(len(meanAdvAccs['hlrgd'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['hlrgd'][i]-meanAdvAccs['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['hlrgd'][i]))))
    for i in range(len(meanAdvAccs['fim'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['fim'][i]-meanAdvAccs['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['fim'][i]))))
    if (noisyTraining):
        for i in range(len(meanAdvAccs['agn'])):
            texts.append(plt.text(advPowersAx[i], meanAdvAccs['agn'][i]-meanAdvAccs['undefended'][i], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['agn'][i]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('aggregated_marg_improv_vs_budget_with_labels.pdf')
    #plt.draw()
    #plt.pause(0.01)

    #noise plots
    #plot noise acc results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.title('AGN accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('noise power ($\sigma$)')
    noisePowersPlot = copy.copy(noisePowers)
    noisePowersPlot.insert(0, 0)
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['hldr'], stdNoiseAccs['hldr'], label='LDR', marker='1')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['kldr'], stdNoiseAccs['kldr'], label=chosenReg, marker='2')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['hlrgd'], stdNoiseAccs['hlrgd'], label='HGD', marker='3')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['fim'], stdNoiseAccs['fim'], label='FIMR', marker='4')
    if (noisyTraining):
        plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['agn'], stdNoiseAccs['agn'], label='AGNT', marker='x')
    plt.legend()
    plt.savefig('aggregatedACCVsBudget_no_labelsAGN.pdf')
    texts = []
    # for i in range(len(meanNoiseAccs['undefended'])):
    #     texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['undefended'][i]))))
    for i in range(len(meanNoiseAccs['hldr'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hldr'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hldr'][i]))))
    for i in range(len(meanNoiseAccs['kldr'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['kldr'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['kldr'][i]))))
    for i in range(len(meanNoiseAccs['hlrgd'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hlrgd'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hlrgd'][i]))))
    for i in range(len(meanNoiseAccs['fim'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['fim'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['fim'][i]))))
    if (noisyTraining):
        for i in range(len(meanNoiseAccs['agn'])):
            texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['agn'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['agn'][i]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('aggregatedACCVsBudget_with_labelsAGN.pdf')
    #plt.draw()
    #plt.pause(0.01)


    #marginal improvement plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.title('adversarial accuracy')
    plt.ylabel('accuracy gain')
    plt.xlabel('noise power ($\sigma$)')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['hldr']-meanNoiseAccs['undefended'], stdNoiseAccs['hldr'], label='LDR', marker='1')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['kldr']-meanNoiseAccs['undefended'], stdNoiseAccs['kldr'], label=chosenReg, marker='2')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['hlrgd']-meanNoiseAccs['undefended'], stdNoiseAccs['hlrgd'], label='HGD', marker='3')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['fim'] - meanNoiseAccs['undefended'], stdNoiseAccs['fim'], label='FIMR', marker='4')
    if (noisyTraining):
        plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['agn'] - meanNoiseAccs['undefended'], stdNoiseAccs['agn'], label='AGNT', marker='x')
    plt.legend()
    plt.savefig('aggregated_marg_improv_vs_budget_no_labelsAGN.pdf')
    texts = []
    # for i in range(len(advAccuracy['undefended'])):
    #     texts.append(plt.text(advPowersAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowers[i]), str(advAccuracy['undefended'][i]))))
    for i in range(len(meanNoiseAccs['hldr'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hldr'][i]-meanNoiseAccs['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hldr'][i]))))
    for i in range(len(meanNoiseAccs['kldr'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['kldr'][i]-meanNoiseAccs['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['kldr'][i]))))
    for i in range(len(meanNoiseAccs['hlrgd'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hlrgd'][i]-meanNoiseAccs['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hlrgd'][i]))))
    for i in range(len(meanNoiseAccs['fim'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['fim'][i]-meanNoiseAccs['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['fim'][i]))))
    if (noisyTraining):
        for i in range(len(meanNoiseAccs['agn'])):
            texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['agn'][i]-meanNoiseAccs['undefended'][i], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['agn'][i]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('aggregated_marg_improv_vs_budget_with_labelsAGN.pdf')
    #plt.draw()
    #plt.pause(0.01)

    #perturbational error amplification plotting
    #plot the maes
    # for curLevelIndex in range(len(advPowers)):
    plt.figure()
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseHLDRMAEs, stdAverageLayerwiseHLDRMAEs, marker='1', label='LDR')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseKLDRMAEs, stdAverageLayerwiseKLDRMAEs, marker='2', label=chosenReg)
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseFIMMAEs, stdAverageLayerwiseFIMMAEs, marker='4', label='FIMR')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseHLRGDMAEs, stdAverageLayerwiseHLRGDMAEs, marker='3', label='HGD')
    if (noisyTraining):
        plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNMAEs, stdAverageLayerwiseAGNMAEs, marker='x', label='AGNT')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseUndefMAEs, stdAverageLayerwiseUndefMAEs, marker='2', label='undefended')
    
    plt.ylabel(r'layer-wise perturbation: mean of ($\frac{\vert \mathbf{s}_i(\mathbf{x})-\mathbf{s}_i(\mathbf{x}_a)\vert}{\vert \mathbf{s}_i(\mathbf{x}) \vert} $)')
    plt.xlabel('hidden layer index ($i$)')
    plt.legend()
    plt.savefig('layerwisePerturbationErrors_.pdf')
    #plt.draw()
    #plt.pause(0.01)

    #plot noise maes
    plt.figure()
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNFIMMAEs, stdAverageLayerwiseAGNFIMMAEs, marker='4', label='FIMR')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNHLRGDMAEs, stdAverageLayerwiseAGNHLRGDMAEs, marker='3', label='HGD')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNHLDRMAEs, stdAverageLayerwiseAGNHLDRMAEs, marker='1', label='LDR')
    # plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNKLDRMAEs, stdAverageLayerwiseAGNKLDRMAEs, marker='1', label=chosenReg)
    if (noisyTraining):
        plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNAGNMAEs, stdAverageLayerwiseAGNAGNMAEs, marker='1', linestyle='-.', label='AGNT')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNUndefMAEs, stdAverageLayerwiseAGNUndefMAEs, marker='2', label='undefended')
    plt.ylabel(r'mean cosine similarity between $\mathbf{s}_i(\mathbf{x})$ and $\mathbf{s}_i(\mathbf{x}_a)$ ')
    plt.xlabel('hidden layer index ($i$)')
    plt.legend()
    plt.savefig('layerwisePerturbationErrors_AGN.pdf')
    #plt.draw()
    #plt.pause(0.01)

#if we tested performance throughout the training procedure (i.e., more than once)
#in this case, we're plotting benign and adversarial test accuracy vs noise magnitude in 3 ways:
#1) in a figure for each model, plot mean acc (\pm CI) vs epsilon for multiple epochs
#2) in a figure for each epoch, plot acc vs epsilon for the various models
#3) plotting the original figures from only the final epoch

elif (testPerfVsEpochs):

    #generate type 1 figure
    #type 1 hldr
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.title('adversarial accuracy')
    plt.title('adv robustness vs training epochs: HLDR')
    plt.ylabel('accuracy')
    plt.xlabel('adversarial budget ($\epsilon$)')
    advPowersPlot = copy.copy(advPowers)
    advPowersPlot.insert(0, 0)
    advPowersAx = np.array(advPowersPlot)
    for curEpoch in range(numberOfEpochs):
        plt.errorbar(np.array(advPowersAx), meanAdvAccs['hldr'][:,curEpoch], stdAdvAccs['hldr'][:,curEpoch], label='epoch %s'%str(curEpoch*testBetweenEpochs))
    plt.legend()    
    plt.savefig('aggregatedRobustnessVsEpsvsEpoch_HLDR.pdf')
    #plt.draw()
    #plt.pause(0.01)

    #type 1 kldr
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.title('adversarial accuracy')
    plt.title('adv robustness vs training epochs: KLDR')
    plt.ylabel('accuracy')
    plt.xlabel('adversarial budget ($\epsilon$)')
    advPowersPlot = copy.copy(advPowers)
    advPowersPlot.insert(0, 0)
    advPowersAx = np.array(advPowersPlot)
    for curEpoch in range(numberOfEpochs):
        plt.errorbar(np.array(advPowersAx), meanAdvAccs['kldr'][:,curEpoch], stdAdvAccs['kldr'][:,curEpoch], label='epoch %s'%str(curEpoch*testBetweenEpochs))
    plt.legend()    
    plt.savefig('aggregatedRobustnessVsEpsvsEpoch_KLDR.pdf')
    #plt.draw()
    #plt.pause(0.01)
    
    #type 1 fim
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.title('adversarial accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('adversarial budget ($\epsilon$)')
    plt.title('adv robustness vs training epochs: FIM')
    advPowersAx = np.array(advPowersPlot)
    for curEpoch in range(numberOfEpochs):
        plt.errorbar(np.array(advPowersAx), meanAdvAccs['fim'][:,curEpoch], stdAdvAccs['fim'][:,curEpoch], label='epoch %s'%str(curEpoch*testBetweenEpochs))
    plt.legend()    
    plt.savefig('aggregatedRobustnessVsEpsvsEpoch_FIM.pdf')
    #plt.draw()
    #plt.pause(0.01)

    #type 1 agn
    if (noisyTraining):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plt.title('adversarial accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('adversarial budget ($\epsilon$)')
        plt.title('adv robustness vs training epochs: AGNT')
        advPowersAx = np.array(advPowersPlot)
        for curEpoch in range(numberOfEpochs):
            plt.errorbar(np.array(advPowersAx), meanAdvAccs['agn'][:,curEpoch], stdAdvAccs['agn'][:,curEpoch], label='epoch %s'%str(curEpoch*testBetweenEpochs))
        plt.legend()    
        plt.savefig('aggregatedRobustnessVsEpsvsEpoch_AGN.pdf')
        #plt.draw()
        #plt.pause(0.01)

    #type 1 hlrgd
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.title('adversarial accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('adversarial budget ($\epsilon$)')
    advPowersAx = np.array(advPowersPlot)
    for curEpoch in range(numberOfEpochs):
        plt.errorbar(np.array(advPowersAx), meanAdvAccs['hlrgd'][:,curEpoch], stdAdvAccs['hlrgd'][:,curEpoch], label='epoch %s'%str(curEpoch*testBetweenEpochs))
    plt.legend()    
    plt.savefig('aggregatedRobustnessVsEpsvsEpoch_HLRGD.pdf')
    #plt.draw()
    #plt.pause(0.01)
        
    # #iterate over epochs to generate type 2 figures
    for curEpoch in range(numberOfEpochs):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plt.title('adversarial accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('adversarial budget ($\epsilon$)')
        plt.title('epoch %s '%str(curEpoch))
        advPowersAx = np.array(advPowersPlot)
        plt.errorbar(np.array(advPowersAx), meanAdvAccs['hldr'][:,curEpoch], stdAdvAccs['hldr'][:,curEpoch], label='LDR', linestyle='-.')
        plt.errorbar(np.array(advPowersAx), meanAdvAccs['kldr'][:,curEpoch], stdAdvAccs['kldr'][:,curEpoch], label=chosenReg, linestyle='--')
        plt.errorbar(np.array(advPowersAx), meanAdvAccs['hlrgd'][:,curEpoch], stdAdvAccs['hlrgd'][:,curEpoch], label='HGD', marker='3')
        plt.errorbar(np.array(advPowersAx), meanAdvAccs['fim'][:,curEpoch], stdAdvAccs['fim'][:,curEpoch], label='FIMR', marker='4')
        if (noisyTraining):
            plt.errorbar(np.array(advPowersAx), meanAdvAccs['agn'][:,curEpoch], stdAdvAccs['agn'][:,curEpoch], label='AGNT', linestyle='-.')
        plt.legend()
        plt.savefig('aggregatedRobustnessVsEpsAllModelsEpoch%s.pdf'%str(curEpoch*testBetweenEpochs))
        #plt.draw()
        #plt.pause(0.01)


    #plot type 3 figures
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.title('adversarial accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('adversarial budget ($\epsilon$)')
    advPowersAx = np.array(advPowersPlot)
    plt.errorbar(advPowersAx, meanAdvAccs['hldr'][:,-1], stdAdvAccs['hldr'][:,-1], label='LDR', marker='1', ms=7)
    plt.errorbar(advPowersAx, meanAdvAccs['kldr'][:,-1], stdAdvAccs['kldr'][:,-1], label=chosenReg, marker='2', ms=7)
    plt.errorbar(advPowersAx, meanAdvAccs['hlrgd'][:,-1], stdAdvAccs['hlrgd'][:,-1], label='HGD', marker='3', ms=7)
    plt.errorbar(advPowersAx, meanAdvAccs['fim'][:,-1], stdAdvAccs['fim'][:,-1], label='FIMR', marker='4', ms=7)
    if (noisyTraining):
        plt.errorbar(advPowersAx, meanAdvAccs['agn'][:,-1], stdAdvAccs['agn'][:,-1], label='AGNT', marker='x', linestyle='-.', ms=7)
    plt.ylim((0,1))
    #inset axes should let us zoom in on the very busy region of small \epsilon
    # inset axes....
    axins = ax.inset_axes([0.6, 0.65, 0.33, 0.33])
    axins.errorbar(advPowersAx, meanAdvAccs['hldr'][:,-1], stdAdvAccs['hldr'][:,-1], label='LDR', marker='1', ms=7)
    axins.errorbar(advPowersAx, meanAdvAccs['kldr'][:,-1], stdAdvAccs['kldr'][:,-1], label=chosenReg, marker='2', ms=7)
    axins.errorbar(advPowersAx, meanAdvAccs['hlrgd'][:,-1], stdAdvAccs['hlrgd'][:,-1], label='HGD', marker='3', ms=7)
    axins.errorbar(advPowersAx, meanAdvAccs['fim'][:,-1], stdAdvAccs['fim'][:,-1], label='FIMR', marker='4', ms=7)
    if (noisyTraining):
        axins.errorbar(advPowersAx, meanAdvAccs['agn'][:,-1], stdAdvAccs['agn'][:,-1], label='AGNT', marker='x', linestyle='-.', ms=7)
    # sub region of the original image
    x1, x2, y1, y2 = -0.0005,0.0127, 0.76, 0.96
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels(np.round(np.linspace(y1, y2, 5),decimals=2), fontdict={'fontsize': 8})

    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.legend()
    # plt.show()
    plt.savefig('aggregatedACCVsBudget_no_labels.pdf')
    
    texts = []
    for i in range(len(meanAdvAccs['undefended'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['undefended'][i,-1], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['undefended'][i,-1]))))
    for i in range(len(meanAdvAccs['hldr'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['hldr'][i,-1], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['hldr'][i,-1]))))
    for i in range(len(meanAdvAccs['kldr'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['kldr'][i,-1], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['kldr'][i,-1]))))
    for i in range(len(meanAdvAccs['hlrgd'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['hlrgd'][i,-1], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['hlrgd'][i,-1]))))
    for i in range(len(meanAdvAccs['fim'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['fim'][i,-1], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['fim'][i,-1]))))
    if (noisyTraining):
        for i in range(len(meanAdvAccs['agn'])):
            texts.append(plt.text(advPowersAx[i], meanAdvAccs['agn'][i,-1], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['agn'][i,-1]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('aggregatedACCVsBudget_with_labels.pdf')
    #plt.draw()
    #plt.pause(0.01)


    #marginal improvement plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylabel('accuracy gain')
    plt.xlabel('adversarial budget ($\epsilon$)')
    plt.errorbar(np.array(advPowersPlot), meanAdvAccs['hldr'][:,-1]-meanAdvAccs['undefended'][:,-1], stdAdvAccs['hldr'][:,-1], label='LDR', marker='1')
    plt.errorbar(np.array(advPowersPlot), meanAdvAccs['kldr'][:,-1]-meanAdvAccs['undefended'][:,-1], stdAdvAccs['kldr'][:,-1], label=chosenReg, marker='1')
    plt.errorbar(np.array(advPowersPlot), meanAdvAccs['hlrgd'][:,-1]-meanAdvAccs['undefended'][:,-1], stdAdvAccs['hlrgd'][:,-1], label='HGD', marker='3')
    plt.errorbar(np.array(advPowersPlot), meanAdvAccs['fim'][:,-1] - meanAdvAccs['undefended'][:,-1], stdAdvAccs['fim'][:,-1], label='FIMR', marker='4')
    if (noisyTraining):
        plt.errorbar(np.array(advPowersPlot), meanAdvAccs['agn'][:,-1] - meanAdvAccs['undefended'][:,-1], stdAdvAccs['agn'][:,-1], label='AGNT', marker='x', linestyle='-.')
    plt.legend()
    plt.savefig('aggregated_marg_improv_vs_budget_no_labels.pdf')
    texts = []
    # for i in range(len(advAccuracy['undefended'])):
    #     texts.append(plt.text(advPowersAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowers[i]), str(advAccuracy['undefended'][i]))))
    for i in range(len(meanAdvAccs['hldr'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['hldr'][i,-1]-meanAdvAccs['undefended'][i,-1], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['hldr'][i,-1]))))
    for i in range(len(meanAdvAccs['hlrgd'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['hlrgd'][i,-1]-meanAdvAccs['undefended'][i,-1], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['hlrgd'][i,-1]))))
    for i in range(len(meanAdvAccs['fim'])):
        texts.append(plt.text(advPowersAx[i], meanAdvAccs['fim'][i,-1]-meanAdvAccs['undefended'][i,-1], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['fim'][i,-1]))))
    if (noisyTraining):
        for i in range(len(meanAdvAccs['agn'])):
            texts.append(plt.text(advPowersAx[i], meanAdvAccs['agn'][i,-1]-meanAdvAccs['undefended'][i,-1], '(%s, %s)'%(str(advPowersPlot[i]), str(meanAdvAccs['agn'][i,-1]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('aggregated_marg_improv_vs_budget_with_labels.pdf')
    #plt.draw()
    #plt.pause(0.01)

    #noise plots
    #plot noise acc results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    noisePowersPlot = copy.copy(noisePowers)
    noisePowersPlot.insert(0, 0)
    # plt.title('AGN accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('noise power ($\sigma$)')
    noisePowersPlot = copy.copy(noisePowers)
    noisePowersPlot.insert(0, 0)
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['hldr'][:,-1], stdNoiseAccs['hldr'][:,-1], label='LDR', marker='1')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['kldr'][:,-1], stdNoiseAccs['kldr'][:,-1], label=chosenReg, marker='2')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['hlrgd'][:,-1], stdNoiseAccs['hlrgd'][:,-1], label='HGD', marker='3')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['fim'][:,-1], stdNoiseAccs['fim'][:,-1], label='FIMR', marker='4')
    if (noisyTraining):
        plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['agn'][:,-1], stdNoiseAccs['agn'][:,-1], label='AGNT', marker='x', linestyle='-.')
    plt.legend()
    plt.savefig('aggregatedACCVsBudget_no_labelsAGN.pdf')
    texts = []
    # for i in range(len(meanNoiseAccs['undefended'])):
    #     texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['undefended'][i,-1], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['undefended'][i,-1]))))
    for i in range(len(meanNoiseAccs['hldr'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hldr'][i,-1], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hldr'][i,-1]))))
    for i in range(len(meanNoiseAccs['hlrgd'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hlrgd'][i,-1], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hlrgd'][i,-1]))))
    for i in range(len(meanNoiseAccs['fim'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['fim'][i,-1], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['fim'][i,-1]))))
    if (noisyTraining):
        for i in range(len(meanNoiseAccs['agn'])):
            texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['agn'][i,-1], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['agn'][i,-1]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('aggregatedACCVsBudget_with_labelsAGN.pdf')
    #plt.draw()
    #plt.pause(0.01)



    #marginal improvement plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.title('adversarial accuracy')
    plt.ylabel('accuracy gain')
    plt.xlabel('noise power ($\sigma$)')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['hldr'][:,-1]-meanNoiseAccs['undefended'][:,-1], stdNoiseAccs['hldr'][:,-1], label='LDR', linestyle='-.')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['kldr'][:,-1]-meanNoiseAccs['undefended'][:,-1], stdNoiseAccs['kldr'][:,-1], label=chosenReg, linestyle='--')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['hlrgd'][:,-1]-meanNoiseAccs['undefended'][:,-1], stdNoiseAccs['hlrgd'][:,-1], label='HGD', marker='3')
    plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['fim'][:,-1] - meanNoiseAccs['undefended'][:,-1], stdNoiseAccs['fim'][:,-1], label='FIMR', marker='4')
    if (noisyTraining):
        plt.errorbar(np.array(noisePowersPlot), meanNoiseAccs['agn'][:,-1] - meanNoiseAccs['undefended'][:,-1], stdNoiseAccs['agn'][:,-1], label='AGNT', marker='x', linestyle='-.')
    plt.legend()
    plt.savefig('aggregated_marg_improv_vs_budget_no_labelsAGN.pdf')
    texts = []
    # for i in range(len(advAccuracy['undefended'])):
    #     texts.append(plt.text(advPowersAx[i], advAccuracy['undefended'][i], '(%s, %s)'%(str(advPowers[i]), str(advAccuracy['undefended'][i]))))
    for i in range(len(meanNoiseAccs['hldr'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hldr'][i,-1]-meanNoiseAccs['undefended'][i,-1], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hldr'][i,-1]))))
    for i in range(len(meanNoiseAccs['hlrgd'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['hlrgd'][i,-1]-meanNoiseAccs['undefended'][i,-1], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['hlrgd'][i,-1]))))
    for i in range(len(meanNoiseAccs['fim'])):
        texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['fim'][i,-1]-meanNoiseAccs['undefended'][i,-1], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['fim'][i,-1]))))
    if (noisyTraining):
        for i in range(len(meanNoiseAccs['agn'])):
            texts.append(plt.text(noisePowersPlot[i], meanNoiseAccs['agn'][i,-1]-meanNoiseAccs['undefended'][i,-1], '(%s, %s)'%(str(noisePowersPlot[i]), str(meanNoiseAccs['agn'][i,-1]))))
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.savefig('aggregated_marg_improv_vs_budget_with_labelsAGN.pdf')
    #plt.draw()
    #plt.pause(0.01)


    #perturbational error amplification plotting
    #plot the maes
    # for curLevelIndex in range(len(advPowers)):
    plt.figure()
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseHLDRMAEs, stdAverageLayerwiseHLDRMAEs, marker='1', label='LDR')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseKLDRMAEs, stdAverageLayerwiseKLDRMAEs, marker='2', label=chosenReg)
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseHLRGDMAEs, stdAverageLayerwiseHLRGDMAEs, marker='3', label='HGD')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseFIMMAEs, stdAverageLayerwiseFIMMAEs, marker='4', label='FIMR')
    if (noisyTraining):
        plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNMAEs, stdAverageLayerwiseAGNMAEs, marker='x', linestyle='-.', label='AGNT')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseUndefMAEs, stdAverageLayerwiseUndefMAEs, marker='.', label='undefended')
    plt.ylabel(r'mean cosine similarity between $\mathbf{s}_i(\mathbf{x})$ and $\mathbf{s}_i(\mathbf{x}_a)$ ')
    plt.xlabel('hidden layer index ($i$)')
    plt.legend()
    plt.savefig('layerwisePerturbationErrors_.pdf')
    #plt.draw()
    #plt.pause(0.01)

    #plot noise maes
    plt.figure()
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNHLDRMAEs, stdAverageLayerwiseAGNHLDRMAEs, marker='1', label='LDR')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNKLDRMAEs, stdAverageLayerwiseAGNKLDRMAEs, marker='2', label=chosenReg)
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNHLRGDMAEs, stdAverageLayerwiseAGNHLRGDMAEs, marker='3', label='HGD')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNFIMMAEs, stdAverageLayerwiseAGNFIMMAEs, marker='4', label='FIMR')
    if (noisyTraining):
        plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNAGNMAEs, stdAverageLayerwiseAGNAGNMAEs, marker='x', linestyle='-.', label='AGNT')
    plt.errorbar(range(numberOfLayers), meanAverageLayerwiseAGNUndefMAEs, stdAverageLayerwiseAGNUndefMAEs, marker='.', label='undefended')
    plt.ylabel(r'mean cosine similarity between $\mathbf{s}_i(\mathbf{x})$ and $\mathbf{s}_i(\mathbf{x}_a)$ ')
    plt.xlabel('hidden layer index ($i$)')
    plt.legend()
    plt.savefig('layerwisePerturbationErrors_AGN.pdf')
    #plt.draw()
    #plt.pause(0.01)
    
    

########################################################################################################################


#prevent program close until figures are closed
# plt.show()