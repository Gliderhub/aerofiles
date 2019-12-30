import time
from .olc import Scorer
import numpy as np
from math import radians
import datetime as dt

def timeit(n):
    scorer = Scorer()
    scorer.import_perlan_flight()
    latlon = np.radians(np.column_stack([scorer.lat, scorer.lon]))
    dist_matrix = scorer.simple_dist_matrix(latlon)

    start_time = time.time()
    for i in range(n):
        graph = scorer.find_graph(dist_matrix)
        path = scorer.find_path(graph, dist_matrix)
        print(scorer.find_distance(path))
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
        latlon = np.column_stack([np.radians(scorer.lat), np.radians(scorer.lon)])
        dist_matrix = scorer.simple_dist_matrix(latlon)
        print(np.shape(dist_matrix))
    print("Simpler dist matrix %s seconds ---" % ((time.time() - start_time)/n))

timeit(5)
