import time
from .olc import Scorer
import numpy as np
from math import radians
import datetime as dt

def timeit(n):
    scorer = Scorer()
    scorer.import_longer_flight()

    start_time = time.time()
    for i in range(n):
        latlon = np.transpose(np.vstack([np.radians(scorer.lat), np.radians(scorer.lon)]))
        dist_matrix = scorer.dist_matrix(latlon)
    print("Dist matrix %s seconds ---" % ((time.time() - start_time)/n))

    start_time = time.time()
    for i in range(n):
        graph, index_graph = scorer.find_graph(dist_matrix)
        path = scorer.find_path(index_graph, reverse_from=np.argmax(graph[:,scorer.layers-1]))
    print("Not vectorized %s seconds ---" % ((time.time() - start_time)/n))

    start_time = time.time()
    for i in range(n):
        graph, index_graph = scorer.find_graph_vectorized(dist_matrix)
        path = scorer.find_path(index_graph, reverse_from=np.argmax(graph[:,scorer.layers-1]))
    print("Vectorized %s seconds ---" % ((time.time() - start_time)/n))

def test_dist_matrix(n):
    scorer = Scorer()
    scorer.import_simple_flight()

    start_time = time.time()
    for i in range(n):
        latlon = np.transpose(np.vstack([np.radians(scorer.lat), np.radians(scorer.lon)]))
        dist_matrix = scorer.haversine_dist_matrix(latlon)
    print("Haversine matrix %s seconds ---" % ((time.time() - start_time)/n))

    start_time = time.time()
    for i in range(n):
        latlon = np.transpose(np.vstack([np.radians(scorer.lat), np.radians(scorer.lon)]))
        simple_dist_matrix = scorer.simple_dist_matrix(latlon)
    print("Simpler dist matrix %s seconds ---" % ((time.time() - start_time)/n))

    start_time = time.time()
    for i in range(n):
        graph, index_graph = scorer.find_graph_vectorized(dist_matrix)
        path1 = scorer.find_path(index_graph, reverse_from=np.argmax(graph[:,scorer.layers-1]))
    print("With haversine dist matrix %s seconds ---" % ((time.time() - start_time)/n))

    start_time = time.time()
    for i in range(n):
        graph, index_graph = scorer.find_graph_vectorized(simple_dist_matrix)
        path2 = scorer.find_path(index_graph, reverse_from=np.argmax(graph[:,scorer.layers-1]))
    print("With simple dist matrix %s seconds ---" % ((time.time() - start_time)/n))

    assert(path1==path2)

scorer = Scorer()
scorer.import_height_difference_flight()
path = scorer.score_with_height()
print(scorer.time[path])
print(scorer.alt[path])
