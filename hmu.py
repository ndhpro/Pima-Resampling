import sys
import numpy as np
import pandas as pd
from math import sqrt
from random import choices


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-2):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


def get_margin(train, test_row):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    for train_row, dist in distances:
        if test_row[-2] == train_row[-2]:
            nearesthit = train_row
            break
    for train_row, dist in distances:
        if test_row[-2] != train_row[-2]:
            nearestmiss = train_row
            break
    margin = 1/2 * (euclidean_distance(test_row, nearestmiss) - euclidean_distance(test_row, nearesthit))
    return margin


def hmu(file_name=sys.argv[1]):
    data = pd.read_csv(file_name)
    data['index'] = range(len(data))
    print(len(data))
    N = len(data[data["' Outcome'"]==0])
    p = len(data[data["' Outcome'"]==1])
    while N >= p + 10:
        margin = list()
        for test_row in data.values:
            if test_row[-2] == 0:
                margin.append((test_row[-1], get_margin(data.values, test_row)))
        margin.sort(key=lambda tup: tup[1])
        drop = [m[0] for m in margin[:10]]
        data = data.drop(drop)
        N -= 10
    print(len(data))
    data.iloc[:, :-1].to_csv('hmu.csv', index=None)
    

if __name__ == "__main__":
    hmu()
