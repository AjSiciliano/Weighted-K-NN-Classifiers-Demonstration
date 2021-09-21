import csv
import math
from collections import Counter
import numpy as np
import random
from main import euclidian_distance
import matplotlib.pyplot as plt
import matplotlib.markers

filename = "datasets/iris.data"

# initializing the titles and rows list
fields = []
total_set = [] #first half training, second half testing
normalized_set = []
types = []

def classification(name): 
    #Associate the class name to a number
    if name == "Iris-setosa":
        return 0
    elif name == "Iris-versicolor":
        return 1
    else:
        return 2

def count_correct(v1,v2):
    #Count the number of similarities between v1 and v2

    num_correct = 0

    for i in range(len(v1)):
        
        if (v1[i] == classification(v2[i])): #classify strings numerically
            num_correct += 1

    return num_correct

def normalization(array):
    #Normalize the attributes in respect to eachother in each column
    normalized_list = []

    maximum = np.maximum.reduce(total_set)
    minimum = np.minimum.reduce(total_set)

    for i in range(len(array)):
        e = array[i]

        normalized_list.append((e - minimum[i]) / (maximum[i] - minimum[i]))

    return normalized_list

#method used to plot example graphs and find figures for individual attributes
def plot(actual, prediction, name, figure):

    r = {0:[],1:[],2:[]}

    num_correct = {0:0,1:0,2:0}

    for x in range(len(actual)):
        r[int(classification(actual[x]))].append(prediction[x])

        if(int(classification(actual[x])) == prediction[x]):
            num_correct[classification(actual[x])] += 1

    for x in r:
        if (len(r[x]) != 0):
            print("class = " + str(x) + " percent correct: " + str(num_correct[x]/len(r[x])))

    #if ploting uncomment below lines and uncomment plot lines at very end of code as well

    b = 2

    plt.figure(figure)
    plt.plot(list(range(0,len(r[b]))),[b]*len(r[b]),label = "expected for c = " + str(b),linewidth=3, color = 'hotpink',zorder=1)
    plt.scatter(list(range(0,len(r[b]))),r[b],label = name + " for c = " + str(b), linewidths=1,zorder=2,color = 'black', marker=matplotlib.markers.TICKDOWN)

def weighted_KNN_classification(test_set, element,k):
    total_distances = []
    
    index = 0
    
    for x in test_set:
        total_distances.append([euclidian_distance([x[i] for i in range(len(x))],element), types[mid + index]])
        index += 1

    total_distances.sort()
    total_distances = total_distances[1:]
    # print(total_distances)

    distances_to_k = [total_distances[i] for i in range(k)]
    distances = [total_distances[i][0] for i in range(len(distances_to_k))]

    maximum = max(distances)
    minimum = min(distances)

    class_dict = {0:[],1:[],2:[]}

    for distance in distances_to_k:
        class_dict[(classification(distance[-1]))].append(distance[0])

    #each index is associated with the sum of the weight for that class where class = index
    new_list = [0,0,0]

    for i in class_dict: 
        # print(i)
        for d in class_dict[i]:

            if maximum != minimum:
                new_list[i] += ((maximum - d) / (maximum - minimum))
            else:
                new_list[i] += 1


    return new_list.index(max(new_list))

def unweighted_KNN_classification(test_set, element,k):
    total_distances = []
    index = 0
    
    for x in test_set:
        total_distances.append([euclidian_distance([x[i] for i in range(len(x))],element),  types[mid + index]])
        index += 1

    total_distances.sort()

    total_distances = total_distances[1:]

    distances_to_k = [total_distances[i] for i in range(k)]

    class_dict = {0:[],1:[],2:[]}

    for distance in distances_to_k:
        class_dict[(classification(distance[-1]))].append(distance[0])

    #each index is associated with the number of nodes with that 
    #class where class = index in nearest neighbors

    new_list = {}

    for i in class_dict: 
        #Associate each class with the length of it's respective distance array
        new_list[(i)] = len(class_dict[i])

    return max(new_list, key=new_list.get)

#___________________________IMPLEMENTATION BELOW___________________________

#read and splice the data

#https://www.geeksforgeeks.org/working-csv-files-python/
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
    
    shuffled = []

    # extracting each data row one by one
    for row in csvreader:
        shuffled.append(row)

    random.seed(4) #seed for replication
    random.shuffle(shuffled)

    for row in shuffled:

        floated_row = [float(row[i]) for i in range(len(row)-1)]
        total_set.append(floated_row)
        types.append(row[-1])

    for row in range(len(total_set)):
        normalized_set.append(normalization(total_set[row]))


#___________________________RUNNING BELOW____________________________________

#first half is test
#second half we use classify
mid = len(normalized_set)//2

k = 22


#________ UNWEIGHTED ________


set_of_predictions = []
predictions = []

for x in range(mid):
    predictions.append(unweighted_KNN_classification(normalized_set[mid:], normalized_set[x], k))

set_of_predictions.append(predictions)

total = 0

avg_set_of_predict = [0] * len(types[:mid])

#if needed can run multiple trails, and should automatically average
for x in set_of_predictions:
    # print(x)
    total += count_correct(x, types[:mid])
    for e in range(len(x)):
        avg_set_of_predict[e] += x[e]


#average general prediction, num correct / expected correct, aka number of elements in set
total = total / len(set_of_predictions)

for e in range(len(avg_set_of_predict)):
    avg_set_of_predict[e] = avg_set_of_predict[e] / len(set_of_predictions)

#grab first prediction, with the first k, since only 1 k used in our implementation
#use this to average our predictions
#a little redudant, but can be extrapolated for further use in future
avg_set_of_predict = set_of_predictions[0]

plot(types[:mid], avg_set_of_predict, "unweighted", 0)

print( "Average error rate w/o weights: " + str(1 - (total / len(types[:mid]))) )
print( "Average correct rate w/o weights: " + str((total / len(types[:mid]))) )

#________ WEIGHTED ________

set_of_predictions = []
predictions = []

for x in range(mid):

    predictions.append(weighted_KNN_classification(normalized_set[mid:], normalized_set[x], k))

set_of_predictions.append(predictions)

total = 0

avg_set_of_predict = [0] * len(types[:mid])

#if needed can run multiple trails, and should automatically average
for x in set_of_predictions:
    total += count_correct(x, types[:mid])
    for e in range(len(x)):
        avg_set_of_predict[e] += x[e]

#average general prediction, num correct / expected correct, aka number of elements in set
total = total / len(set_of_predictions)

for e in range(len(avg_set_of_predict)):
    avg_set_of_predict[e] = avg_set_of_predict[e] / len(set_of_predictions)

types_to_num = []

for x in types[:mid]:
    types_to_num.append(classification(x))

#grab first prediction, with the first k, since only 1 k used in our implementation
#use this to average our predictions
#a little redudant, but can be extrapolated for further use in future
avg_set_of_predict = set_of_predictions[0]

plot(types[:mid], avg_set_of_predict, "weighted", 1)

print( "Average error rate w/weights: " + str(1 - (total / len(types[:mid]))) )
print( "Average correct rate w/weights: " + str((total / len(types[:mid]))) )

#if ploting uncomment plot lines in plot function, and uncomment below as well

plt.show()







