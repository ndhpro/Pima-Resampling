import sys
import numpy as np
import pandas as pd
from math import sqrt
from random import choices, random, randint
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-2):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)


def get_xc(train, test_row):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	return distances[0][0]


def get_min_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        if train_row[-2] == 1:
            dist = euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return np.array(neighbors, dtype=np.float)


def dec(file_name=sys.argv[1]):
    data = pd.read_csv(file_name)
    data['index'] = range(len(data))
    print(len(data))
    for test_row in data[data["' Outcome'"] == 1].values:
        neighbors = get_min_neighbors(data.values, test_row, 10)
        chosen = choices(neighbors, k=2)
        mu = test_row[:-2] + random()*(chosen[0][:-2]-chosen[1][:-2])
        de = list()
        for j in range(len(test_row)-2):
            s = randint(0, len(test_row)-2)
            CR = 0.5
            if j == s or random() <= CR:
                de.append(mu[j])
            else:
                de.append(test_row[j])
        de.extend([1, len(data)])
        data = data.append(dict(zip(data.columns, de)), ignore_index=True)
    print(len(data))
    data.iloc[:, :-1].to_csv('dec.csv', index=None)

    kmeans = KMeans().fit(data.iloc[:, :-2].values)
    data['cluster'] = kmeans.labels_
    for c in range(8):
        cluster = data.loc[data['cluster']==c, :'index']
        fl0 = fl1 = 0
        for test_row in cluster.values:
            if test_row[-2] == 0:
                fl0 = 1
            else:
                fl1 = 1
            if fl0 and fl1:
                break
        if not (fl0 and fl1):
            continue
        x = np.mean(cluster.values, axis=0)
        xc = get_xc(cluster.values, x)
        s = 0.9
        for test_row in cluster.values:
            sim = cosine_similarity(test_row[:-2].reshape(1, -1), xc[:-2].reshape(1, -1))
            if sim[0][0] > s:
                data = data.drop([test_row[-1]])
    print(len(data))
    data.iloc[:, :-2].to_csv('dec.csv', index=None)


if __name__ == "__main__":
    dec()
