import time
from .olc import Scorer
import numpy as np
from math import radians
import datetime as dt

def timeit(n):
    scorer = Scorer()
    scorer.import_perlan_flight()

    latlon = np.column_stack([np.radians(self.lat), np.radians(self.lon)])
    start_time = time.time()
    for i in range(n):
        dist_matrix = scorer.simple_dist_matrix(latlon)
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
        latlon = np.column_stack([np.radians(self.lat), np.radians(self.lon)])
        dist_matrix = scorer.haversine_dist_matrix(latlon)
    print("Haversine matrix %s seconds ---" % ((time.time() - start_time)/n))

    start_time = time.time()
    for i in range(n):
        latlon = np.column_stack([np.radians(self.lat), np.radians(self.lon)])
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

def time_douglas_peucker(n, epsilon_meter=500):
    scorer = Scorer()
    scorer.import_simple_flight()

    epsilon = epsilon_meter / 6371000

    start_time = time.time()
    for i in range(n):
        path = scorer.score_with_height(epsilon=epsilon)
    print("With douglas peucker %s seconds ---" % ((time.time() - start_time)/n))
    print(f'Distance: {scorer.find_distance(path)}')

    start_time = time.time()
    for i in range(n):
        path = scorer.score_with_height()
    print("Without douglas peucker %s seconds ---" % ((time.time() - start_time)/n))
    print(f'Distance: {scorer.find_distance(path)}')

def test_small_flight(n):
    scorer = Scorer()
    scorer.alt = [1000] * n
    scorer.lat = list(range(1, 1+n))
    scorer.lon = list(range(1, 1+n))
    print(len(scorer.alt), len(scorer.lat), len(scorer.lon))
    path = scorer.score()
    print(path)

timeit(2)
