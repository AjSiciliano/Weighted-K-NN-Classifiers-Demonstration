import numpy as np
import math
from main import *

def count_correct(v1,v2):
    #Count the number of similarities between v1 and v2

    num_correct = 0

    for i in range(len(v1)):
        
        if (v1[i] == float(v2[i])):
            num_correct += 1

    return num_correct

def euclidian_distance(row1, row2):
    #Calculate euclidian distance between two vectors r1 and r2
    distance = 0

    for features in range(len(row1)):
        distance += math.pow(float(row1[features]) - float(row2[features]), 2)

    return math.pow(distance, 1/2)