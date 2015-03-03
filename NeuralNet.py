import copy
import sys
from datetime import datetime
from math import exp
from random import random, randint, choice

class Perceptron(object):
    """
    Class to represent a single Perceptron in the net.
    """
    def __init__(self, inSize=1, weights=None):
        self.inSize = inSize+1#number of perceptrons feeding into this one; add one for bias
        if weights is None:
            #weights of previous layers into this one, random if passed in as None
            self.weights = [1.0]*self.inSize
            self.setRandomWeights()
        else:
            self.weights = weights
    
    def getWeightedSum(self, inActs):
        """
        Returns the sum of the input weighted by the weights.
        
        Inputs:
            inActs (list<float/int>): input values, same as length as inSize
        Returns:
            float
            The weighted sum
        """
        return sum([inAct*inWt for inAct,inWt in zip(inActs,self.weights)])
    
    def sigmoid(self, value):
        """
        Return the value of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the sigmoid function parametrized by 
            the value.
        """
        """YOUR CODE"""
        return 1/ (1 + exp(-value))
      
    def sigmoidActivation(self, inActs):                                       
        """
        Returns the activation value of this Perceptron with the given input.
        Same as rounded g(z) in book.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The rounded value of the sigmoid of the weighted input
        """
        acts = [1.0] + inActs
        return round(self.sigmoid(self.getWeightedSum(acts)))
        
        
    def sigmoidDeriv(self, value):
        """
        Return the value of the derivative of a sigmoid function.
        
        Args:
            value (float): the value to get sigmoid for
        Returns:
            float
            The output of the derivative of a sigmoid function
            parametrized by the value.
        """
        return exp(value) / (pow((exp(value) + 1), 2))
        
    def sigmoidActivationDeriv(self, inActs):
        """
        Returns the derivative of the activation of this Perceptron with the
        given input. Same as g'(z) in book (note that this is not rounded.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
        Returns:
            int
            The derivative of the sigmoid of the weighted input
        """
        acts = [1.0] + inActs
        return self.sigmoidDeriv(self.getWeightedSum(acts))
    
    def updateWeights(self, inActs, alpha, delta):
        """
        Updates the weights for this Perceptron given the input delta.
        Remember to add 1 to the start of inActs for the bias input.
        
        Inputs:
            inActs (list<float/int>): input values, not including bias
            alpha (float): The learning rate
            delta (float): If this is an output, then g'(z)*error
                           If this is a hidden unit, then the as defined-
                           g'(z)*sum over weight*delta for the next layer
        Returns:
            float
            Return the total modification of all the weights (absolute total)
        """
        totalModification = 0
        """YOUR CODE"""
        acts = [1.0] + inActs
        i = 0
        for weight in self.weights:
            modification = alpha * acts[i] * delta
            self.weights[i] = weight + modification
            totalModification += abs(modification)
            i += 1
        
        return totalModification
            
    def setRandomWeights(self):
        """
        Generates random input weights that vary from -1.0 to 1.0
        """
        for i in range(self.inSize):
            self.weights[i] = (random() + .0001) * (choice([-1,1]))
        
    def __str__(self):
        """ toString """
        outStr = ''
        outStr += 'Perceptron with %d inputs\n'%self.inSize
        outStr += 'Node input weights %s\n'%str(self.weights)
        return outStr

class NeuralNet(object):                                    
    """
    Class to hold the net of perceptrons and implement functions for it.
    """          
    def __init__(self, layerSize):#default 3 layer, 1 percep per layer
        """
        Initiates the NN with the given sizes.
        
        Args:
            layerSize (list<int>): the number of perceptrons in each layer 
        """
        self.layerSize = layerSize #Holds number of inputs and percepetrons in each layer
        self.outputLayer = []
        self.numHiddenLayers = len(layerSize)-2
        self.hiddenLayers = [[] for x in range(self.numHiddenLayers)]
        self.numLayers =  self.numHiddenLayers+1
        
        #build hidden layer(s)        
        for h in range(self.numHiddenLayers):
            for p in range(layerSize[h+1]):
                percep = Perceptron(layerSize[h]) # num of perceps feeding into this one
                self.hiddenLayers[h].append(percep)
 
        #build output layer
        for i in range(layerSize[-1]):
            percep = Perceptron(layerSize[-2]) # num of perceps feeding into this one
            self.outputLayer.append(percep)
            
        #build layers list that holds all layers in order - use this structure
        # to implement back propagation
        self.layers = [self.hiddenLayers[h] for h in xrange(self.numHiddenLayers)] + [self.outputLayer]
  
    def __str__(self):
        """toString"""
        outStr = ''
        outStr +='\n'
        for hiddenIndex in range(self.numHiddenLayers):
            outStr += '\nHidden Layer #%d'%hiddenIndex
            for index in range(len(self.hiddenLayers[hiddenIndex])):
                outStr += 'Percep #%d: %s'%(index,str(self.hiddenLayers[hiddenIndex][index]))
            outStr +='\n'
        for i in range(len(self.outputLayer)):
            outStr += 'Output Percep #%d:%s'%(i,str(self.outputLayer[i]))
        return outStr
    
    def feedForward(self, inActs):
        """
        Propagate input vector forward to calculate outputs.
        
        Args:
            inActs (list<float>): the input to the NN (an example) 
        Returns:
            list<list<float/int>>
            A list of lists. The first list is the input list, and the others are
            lists of the output values 0f all perceptrons in each layer.
        """
        """YOUR CODE"""
        
        output = [inActs]
        
        acts = inActs[:]
        for layer in self.layers:
            newActs = []
            for node in layer:
                newActs.append(node.sigmoidActivation(acts))
            acts = newActs
            output.append(newActs)

        return output
    
    def backPropLearning(self, examples, alpha = 0.1):
        """
        Run a single iteration of backward propagation learning algorithm.
        See the text and slides for pseudo code.
        NOTE : the pseudo code in the book has an error - 
        you should not update the weights while backpropagating; 
        follow the comments below or the description in lecture.
        
        Args: 
            examples (list<tuple<list,list>>):for each tuple first element is input(feature) "vector" (list)
                                                             second element is output "vector" (list)
            alpha (float): the alpha to training with
        Returns
           tuple<float,float>
           
           A tuple of averageError and averageWeightChange, to be used as stopping conditions. 
           averageError is the summed error^2/2 of all examples, divided by numExamples*numOutputs.
           averageWeightChange is the summed weight change of all perceptrons, divided by the sum of 
               their input sizes.
        """
        #keep track of output
        averageError = 0

        averageWeightChange = 0
        numWeights = 0
        outputLayer = self.outputLayer
        hiddenLayers = self.hiddenLayers
        
        for example in examples:
            deltas = []
            
            exampleInput = example[0]
            exampleOutput = example[1]
            
            output = self.feedForward(exampleInput)
            output.reverse()
            
            # get output deltas
            errors = [e - f for e, f in zip(exampleOutput, output[0])]
            averageError += sum([error ** 2 / 2 for error in errors])
            
            deltas.append(self.outputDeltas(errors, output[1]))

            output.reverse()

            for index in range(len(self.hiddenLayers) - 1, -1, -1):
                newDeltas = []
                layer = self.hiddenLayers[index]
                for i in range(len(layer)):
                    node = layer[i]
                    weightedSum = 0.0
                    deltaIndex = 0
                    for perceptron in self.layers[index + 1]:
                        weightedSum += perceptron.weights[i + 1] * deltas[0][deltaIndex]
                        deltaIndex += 1
                    newDeltas.append(node.sigmoidActivationDeriv(output[index]) * weightedSum)
                deltas.insert(0, newDeltas)

            """
            Having aggregated all deltas, update the weights of the 
            hidden and output layers accordingly.
            """
            for index in range(len(self.layers)):
                layer=self.layers[index]
                for i in range(len(layer)):
                    averageWeightChange+=layer[i].updateWeights(output[index],alpha,deltas[index][i])
                    numWeights+=layer[i].inSize
            
        #end for each example
        
        """Calculate final output"""
        averageError = averageError / (len(examples) * len(examples[0][1]))
        averageWeightChange /= numWeights

        
        return averageError, averageWeightChange

    def outputDeltas(self, error, inputs):
        outputDelta = []
        for err, node in zip(error, self.outputLayer):
            delta = node.sigmoidActivationDeriv(inputs)
            delta *= err
            outputDelta.append(delta)
        return outputDelta

    def getNextLayerDeltas(self, oldLayer, backPropDeltas):
        oldDeltas = []
        for node, delta in zip(oldLayer, backPropDeltas):
            perceptronDeltas = []
            weights = node.weights[:]
            del weights[0]
            for weight in weights:
                perceptronDeltas.append(weight * delta)
            oldDeltas.append(perceptronDeltas)
        return oldDeltas

    def getLayerDeltas(self, newLayer, layerInputs, weightSums):
        layerDeltas = []
        for node in newLayer:
            layerDeltas.append(node.sigmoidActivationDeriv(layerInputs) * weightSums[node])
        return layerDeltas
    
    def getWeightedSums(self, newLayer, outputDeltas):
        weightSums = {}
        for perceptronDeltas in outputDeltas:
            for node, delta in zip(newLayer, perceptronDeltas):
                if node in weightSums:
                    weightSums[node] += delta
                else:
                    weightSums[node] = delta
        return weightSums

def buildNeuralNet(examples, alpha=0.1, weightChangeThreshold = 0.00008,hiddenLayerList = [1], maxItr = sys.maxint, startNNet = None):
    """
    Train a neural net for the given input.
    
    Args: 
        examples (tuple<list<tuple<list,list>>,
                        list<tuple<list,list>>>): A tuple of training and test examples
        alpha (float): the alpha to train with
        weightChangeThreshold (float):           The threshold to stop training at
        maxItr (int):                            Maximum number of iterations to run
        hiddenLayerList (list<int>):             The list of numbers of Perceptrons 
                                                 for the hidden layer(s). 
        startNNet (NeuralNet):                   A NeuralNet to train, or none if a new NeuralNet
                                                 can be trained from random weights.
    Returns
       tuple<NeuralNet,float>
       
       A tuple of the trained Neural Network and the accuracy that it achieved 
       once the weight modification reached the threshold, or the iteration 
       exceeds the maximum iteration.
    """
    examplesTrain,examplesTest = examples       
    numIn = len(examplesTrain[0][0])
    numOut = len(examplesTest[0][1])     
    time = datetime.now().time()
    if startNNet is not None:
        hiddenLayerList = [len(layer) for layer in startNNet.hiddenLayers]
    print "Starting training at time %s with %d inputs, %d ouputs, %s hidden layers, size of training set %d, and size of test set %d"\
                                                    %(str(time),numIn,numOut,str(hiddenLayerList),len(examplesTrain),len(examplesTest))
    layerList = [numIn]+hiddenLayerList+[numOut]
    nnet = NeuralNet(layerList)                                                    
    if startNNet is not None:
        nnet =startNNet
    """
    YOUR CODE
    """
    iteration=0
    trainError=0
    weightMod=1
    
    """
    Iterate for as long as it takes to reach weight modification threshold
    """
    while (iteration < maxItr) and (weightMod > weightChangeThreshold):
        trainError, weightMod = nnet.backPropLearning(examplesTrain, alpha)
        iteration += 1

        #if iteration%10==0:
        #    print '! on iteration %d; training error %f and weight change %f'%(iteration,trainError,weightMod)
        #else :
        #    print '.',
          
    time = datetime.now().time()
    print 'Finished after %d iterations at time %s with training error %f and weight change %f'%(iteration,str(time),trainError,weightMod)
                
    """
    Get the accuracy of your Neural Network on the test examples.
    """ 
    
    testError = 0
    testGood = 0

    for example in examplesTest:
        output = nnet.feedForward(example[0])
        if output[-1] == example[1]:
            testGood += 1
        else:
            testError += 1
    
    testAccuracy = testGood / float(len(examplesTest))
    
    print 'Feed Forward Test correctly classified %d, incorrectly classified %d, test percent error  %f\n'%(testGood,testError,testAccuracy)
    
    """return something"""
    return nnet, testAccuracy

