import time
from .olc import Scorer
import numpy as np
from math import radians
import datetime as dt

def timeit(n):
    scorer = Scorer()

    start_time = time.time()
    for i in range(n):
        scorer.import_torben_flight()
        path = scorer.score_with_height_backwards()
    print("New %s seconds ---" % ((time.time() - start_time)/n))

    print(path)
    print(scorer.find_distance(path))
    print([scorer.alt[p] for p in path])
    print([scorer.time[p] for p in path])

def test_dist_matrix(n):
    scorer = Scorer()
    scorer.import_perlan_flight()

    start_time = time.time()
    for i in range(n):
        latlon = np.column_stack([np.radians(scorer.lat), np.radians(scorer.lon)])
        dist_matrix = scorer.simple_dist_matrix(latlon)
        print(np.shape(dist_matrix))
    print("Simpler dist matrix %s seconds ---" % ((time.time() - start_time)/n))

    start_time = time.time()
    for i in range(n):
        scorer.score_with_height()
    print("Scoring %s seconds ---" % ((time.time() - start_time)/n))

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

timeit(1)
