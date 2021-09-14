import csv
import math
from collections import Counter

filename = "winequality-red.csv"

# initializing the titles and rows list
fields = []
rows = []
qualities = []
def normalization(array):
    mx = max(array)
    mn = min(array)
    r = []
    for element in array:
        e = element
        r.append((e - mn) / (mx - mn))

    return r

#https://www.geeksforgeeks.org/working-csv-files-python/
with open(filename, 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)
      
    # extracting field names through first row
    fields = next(csvreader)
  
    # extracting each data row one by one
    for row in csvreader:
        spliced_row = row[0].split(";")
        floated_row = [float(i) for i in spliced_row]
        qualities.append(floated_row.pop())
        rows.append(normalization(floated_row))

# print(normalization([7,4,25,-5,10]))

def euclidian_distance(row1, row2):
    s = 0

    for c in range(len(row1)):
        s += math.pow(float(row1[c]) - float(row2[c]), 2)

    return math.pow(s, 1/2)

def nearest_neighbors(element, k):
    distances = []
    c = 0
    for x in rows:
        #deal with duplicate nodes
        distances.append([euclidian_distance([x[i] for i in range(len(x)-1)],element), c,  qualities[c]])
        c += 1

    distances.sort()
    print([distances[i] for i in range(k)])

    return [distances[i][2] for i in range(k)]

NN = nearest_neighbors(rows[8], 5)

print(Counter(NN).most_common(1))

# print(rows[8])


# print(rows[0])
# print(normalization(rows[0]))
# print(euclidian_distance(rows[0], rows[1]))
