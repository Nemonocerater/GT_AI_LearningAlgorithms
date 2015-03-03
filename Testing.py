from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt

def average(list):
    return sum(list)/float(len(list))

def stDeviation(list):
    mean = average(list)
    diffSq = [pow((val-mean),2) for val in list]
    return sqrt(sum(diffSq)/len(list))

penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)

def q5(func):
	acc = []
	for i in range(5):
		nnet, accuracy = testPenData()
		acc.append(accuracy)
		
	print "~~~~~~~~~~DATA~~~~~~~~~~~~~~~~~~~~~~"
	print "Max:", max(acc)
	print "Average:", average(acc)
	print "Standard Deviation:", stDeviation(acc)

#q5(testPenData)
#q5(testCarData)

def q6(function):
	acc = {}
	for numPercepts in range(0, 41, 5):
		acc[numPercepts] = []
		for i in range(0,5):
			nnet, error = function([numPercepts])
			acc[numPercepts].append(error)

        print acc
	print "~~~~~~~~~~~~~~~~~~~~DATA~~~~~"
	for numPercepts in range(0, 41, 5):
                runs = acc[numPercepts]
		print numPercepts
		print "   Max:", max(runs)
		print "   Average:", average(runs)
		print "   Standard Deviation:", stDeviation(runs)

q6(testPenData)
#q6(testCarData)
