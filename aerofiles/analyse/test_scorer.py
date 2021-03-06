import time
from .score import Scorer
import numpy as np
from math import radians
import datetime as dt
import scipy
import sklearn
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import euclidean_distances



def compare_backward_forward(n):
    scorer = Scorer()
    scorer.import_torben_flight()
    latlon = np.radians(np.column_stack([scorer.lat, scorer.lon]))
    start_time = time.time()
    for i in range(n):
        path = scorer.score_with_height()
    print("New %s seconds ---" % ((time.time() - start_time)/n))

    print(path)
    print(scorer.find_distance(path))
    print([scorer.alt[p] for p in path])
    print([str(scorer.time[p]) for p in path])

    start_time = time.time()
    for i in range(n):
        path = scorer.score_with_height_backwards()
    print("New %s seconds ---" % ((time.time() - start_time)/n))

    print(path)
    print(scorer.find_distance(path))
    print([scorer.alt[p] for p in path])
    print([str(scorer.time[p]) for p in path])

def test_dist_matrix(n):
    scorer = Scorer()
    scorer.import_perlan_flight()

    start_time = time.time()
    for i in range(n):
        latlon = np.column_stack([np.radians(scorer.lat), np.radians(scorer.lon)]).astype('float32')
        theta = np.cos(np.mean(latlon[:,0]))
        latlon[:,1] *= theta
        matrix1 = scipy.spatial.distance.cdist(latlon, latlon, 'euclidean')
    print("Float32 matrix %s seconds ---" % ((time.time() - start_time)/n))

    start_time = time.time()
    for i in range(n):
        latlon = np.column_stack([np.radians(scorer.lat), np.radians(scorer.lon)]).astype('float64')
        theta = np.cos(np.mean(latlon[:,0]))
        latlon[:,1] *= theta
        matrix1 = scipy.spatial.distance.cdist(latlon, latlon, 'euclidean')
    print("Float64 dist matrix %s seconds ---" % ((time.time() - start_time)/n))

    start_time = time.time()
    for i in range(n):
        latlon = np.column_stack([np.radians(scorer.lat), np.radians(scorer.lon)]).astype('float16')
        theta = np.cos(np.mean(latlon[:,0]))
        latlon[:,1] *= theta
        matrix1 = scipy.spatial.distance.cdist(latlon, latlon, 'euclidean')
    print("Float16 dist matrix %s seconds ---" % ((time.time() - start_time)/n))


    # start_time = time.time()
    # for i in range(n):
    #     condensed = scipy.spatial.distance.pdist(latlon, 'euclidean')
    #     print(np.shape(condensed))
    # print("Simpler dist matrix %s seconds ---" % ((time.time() - start_time)/n))
    #
    # start_time = time.time()
    # for i in range(n):
    #     N = np.shape(latlon)[0]
    #     a,b = np.triu_indices(N,k=1)
    #
    #     # Fill distance matrix
    #     dist_matrix = np.zeros((N,N))
    #     for i in range(len(condensed)):
    #         dist_matrix[a[i],b[i]] = condensed[i]
    #         dist_matrix[b[i],a[i]] = condensed[i]
    # print("Triangular %s seconds ---" % ((time.time() - start_time)/n))

    # start_time = time.time()
    # for i in range(n):
    #     matrix3 = scipy.spatial.distance.squareform(condensed)
    #     print(np.shape(matrix3)[0]**2)
    # print("Cdist matrix %s seconds ---" % ((time.time() - start_time)/n))
    # print(condensed[10000])
    # print(matrix3[100, 100])

def time_dist_matrix(n):
    scorer = Scorer()
    scorer.import_perlan_flight()
    latlon = np.column_stack([np.radians(scorer.lat), np.radians(scorer.lon)])
    theta = np.cos(np.mean(latlon[:,0]))
    latlon[:,1] *= theta
    start_time = time.time()
    for i in range(n):
        res = euclidean_distances(latlon, latlon)
    print("Simpler dist matrix %s seconds ---" % ((time.time() - start_time)/n))


time_dist_matrix(2)
