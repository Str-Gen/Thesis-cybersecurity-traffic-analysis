#! /usr/bin/python2
import csv
import random
import math
import operator

def loadDataset(filename, split, trainingSet=[], testSet=[]):
    # open a csv file
    with open(filename,'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            # add row to training or test depending on the outcome of the random compared to the split
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

# data1 = [2,2,2,'a']
# data2 = [4,4,4,'b']
# distance = euclideanDistance(data1,data2,3)
# print 'Distance: ' + repr(distance)

def manhattenDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += abs((instance1[x]-instance2[x]))
    return distance

# data1 = [2,2,2,'a']
# data2 = [4,4,4,'b']
# distance = euclideanDistance(data1,data2,3)
# print 'Distance: ' + repr(distance)

def getNeighbors(trainingSet,testInstance,k,distanceFunc):
    distances = []
    # how many attributes does the test instance have?
    length = len(testInstance) -1
    # create an index x over the trainingSet items
    for x in range(len(trainingSet)):
        # measure distance between testInstance and the trainingSet at x, for length attr
        dist = distanceFunc(testInstance,trainingSet[x],length)
        # append a list of both the tested trainingSet item and the distance
        distances.append((trainingSet[x],dist))
    # sort distances on 2nd column = dist 
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        # get the k highest ranked instances (most similar = smallest distance)
        neighbors.append(distances[x][0])
    return neighbors

# trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
# testInstance = [5, 5, 5]
# k = 1
# neighbors = getNeighbors(trainSet, testInstance, 1)
# print(neighbors)


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        # the neighbors carry their class as the final element of their feature array
        # we take that as a response, if it doesn't exist we add it to the list of possible responses
        # if the class has been seen before we add a vote
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    # we sort the classvotes object, it's a collection of pairs, 1: class, 2: votes
    # we sort descendingly on the number of votes and let the democratic process run, providing an elected class
    sortedVotes = sorted(classVotes.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

# neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
# response = getResponse(neighbors)
# print(response)

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        # the samples are labeled with the proper category and stored as the final element of the feature array
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

# testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
# predictions = ['a', 'a', 'a']
# accuracy = getAccuracy(testSet, predictions)
# print(accuracy)


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.67
    loadDataset('iris.dat',split,trainingSet,testSet)
    print 'Train set: ' + repr(len(trainingSet))
    print 'Test set: ' + repr(len(testSet))
    # predict
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet,testSet[x],k,manhattenDistance)
        result = getResponse(neighbors)
        predictions.append(result)
        print ('> predicted=' + repr(result) + ', actual='+repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet,predictions)
    print ('Accuracy: '+repr(accuracy) + '%')

main()