import os
import numpy as np
import scipy
import datetime
from sklearn.neighbors import DistanceMetric
from math import radians

from aerofiles.igc import Reader
from rdp import rdp


class Scorer:
    """
    Find polygonal line of maximal length with data points as vertices.
    Height difference between starting point and end point is maximal 1000m.
    """

    def __init__(self):
        self.layers = 6
        self.test_data_dir = 'aerofiles/score/test_data'

    def import_torben_flight(self):
        tow_release = datetime.time(9, 2, 0)

        test_file = os.path.join(self.test_data_dir, '87ilqqk1.igc')
        with open(test_file, 'r') as f:
            parsed = Reader().read(f)
        records = parsed['fix_records'][1]
        for i, record in enumerate(records):
            if record['time'] >= tow_release:
                tow_release_index = i
                break

        records = records[tow_release_index:]
        self.lat = np.array([r['lat'] for r in records])
        self.lon = np.array([r['lon'] for r in records])
        self.time = np.array([r['time'] for r in records])
        self.alt = np.array([r['pressure_alt'] for r in records])

    def import_height_difference_flight(self):
        tow_release = datetime.time(8, 28, 48)
        engine_start = datetime.time(15, 34, 0)

        test_file = os.path.join(self.test_data_dir, '85cd7pd1.igc')
        with open(test_file, 'r') as f:
            parsed = Reader().read(f)
        records = parsed['fix_records'][1]
        for i, record in enumerate(records):
            if record['time'] == tow_release:
                tow_release_index = i
                break
        for i, record in enumerate(records):
            if record['time'] > engine_start:
                engine_start_index = i
                break

        records = records[tow_release_index:engine_start_index]
        self.lat = np.array([r['lat'] for r in records])
        self.lon = np.array([r['lon'] for r in records])
        self.time = np.array([r['time'] for r in records])
        self.alt = np.array([r['pressure_alt'] for r in records])

    def import_longer_flight(self):
        tow_release = datetime.time(8, 42, 0)

        test_file = os.path.join(self.test_data_dir, '87jv20o1.igc')
        with open(test_file, 'r') as f:
            parsed = Reader().read(f)
        records = parsed['fix_records'][1]
        for i, record in enumerate(records):
            if record['time'] > tow_release:
                tow_release_index = i
                break

        records = records[tow_release_index:]
        self.lat = np.array([r['lat'] for r in records])
        self.lon = np.array([r['lon'] for r in records])
        self.alt = np.array([r['pressure_alt'] for r in records])
        self.time = np.array([r['time'] for r in records])

    def import_perlan_flight(self):
        tow_release = datetime.time(16, 54, 10)

        test_file = os.path.join(self.test_data_dir, '99bv7r92.igc')
        with open(test_file, 'r') as f:
            parsed = Reader().read(f)
        records = parsed['fix_records'][1]
        for i, record in enumerate(records):
            if record['time'] > tow_release:
                tow_release_index = i
                break

        records = records[tow_release_index:]
        self.lat = np.array([r['lat'] for r in records])
        self.lon = np.array([r['lon'] for r in records])
        self.alt = np.array([r['pressure_alt'] for r in records])
        self.time = np.array([r['time'] for r in records])

    def import_simple_flight(self):
        test_file = os.path.join(self.test_data_dir, '825lqkk1.igc')
        with open(test_file, 'r') as f:
            parsed = Reader().read(f)

        records = parsed['fix_records'][1]
        self.lat = np.array([r['lat'] for r in records])
        self.lon = np.array([r['lon'] for r in records])
        self.alt = np.array([r['pressure_alt'] for r in records])
        self.time = np.array([r['time'] for r in records])

    def simple_dist_matrix(self, latlon):
        # latlon.shape (10000,2)
        theta = np.cos(np.mean(latlon[:,0]))
        latlon[:,1] *= theta

        condensed = scipy.spatial.distance.pdist(latlon, 'euclidean')
        return scipy.spatial.distance.squareform(condensed)

    def find_graph(self, dist_matrix, fake_dist_matrix=None):
        """
        Calculates (k,l) shaped graph where k is the number of knots
        (data points) and l is the number of layers or legs.
        Graph is used to store the optimum distance that can be achieved with
        l layers and knot k.
        The index graph stores the indices of the previous knot.
        Fake_dist_matrix is used to only allow certain start indices that obey
        the height constrain.
        """
        knots = np.shape(dist_matrix)[0]

        graph = np.zeros((self.layers,knots))
        index_graph = np.zeros((self.layers,knots), dtype='int32')

        # copy reference to used dist_matrix for init (no extra storage)
        if fake_dist_matrix is not None:
            fake_dist_matrix = fake_dist_matrix.T
            for k in range(0, knots):
                index_graph[0,k] = np.argmax(fake_dist_matrix[k,:k+1])
                graph[0,k] = fake_dist_matrix[k,:k+1][index_graph[0,k]]
        else:
            for k in range(0, knots):
                index_graph[0,k] = np.argmax(dist_matrix[k,:k+1])
                graph[0,k] = dist_matrix[k,:k+1][index_graph[0,k]]

        # iterating every layer is vectorized
        for k in range(0, knots):
            options_graph = (graph[:self.layers-1,:k+1] +
                np.expand_dims(dist_matrix[k,:k+1], axis=0))
            index_graph[1:,k] = np.argmax(options_graph, axis=1)
            row_idx = np.arange(np.shape(options_graph)[0])
            graph[1:,k] = options_graph[row_idx,index_graph[1:,k]]

        return graph.T, index_graph.T


    def find_path(self, index_graph, reverse_from):
        """
        Calculates (k,l) shaped graph where k is the number of knots
        (data points) and l is the number of layers or legs.
        Graph is used to store the optimum distance that can be achieved with
        l layers and knot k.
        The index graph stores the indices of the previous knot.
        """
        path = [reverse_from]

        for l in reversed(range(self.layers)):
            path.append(index_graph[path[-1],l])

        return list(reversed(path))

    def distance_from_graph(self, graph):
        return np.max(graph[:,self.layers-1])

    def find_distance(self, path):
        """
        We don't store actual distances in the graph anymore. Therefore the
        distance needs to be calculated from the indexes
        """
        def haversine(lat1, lon1, lat2, lon2):
            """
            Calculate the great circle distance between two points
            on the earth (specified in decimal degrees)
            """
            from math import radians, cos, sin, asin, sqrt
            lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
            # calculate haversine
            lat = lat2 - lat1
            lon = lon2 - lon1
            d = sin(lat * 0.5) ** 2 + cos(lat1) * cos(lat2) * sin(lon * 0.5) ** 2

            return 2 * 6371 * asin(sqrt(d))

        total_distance = 0
        for p1, p2 in zip(path, path[1:]):
            total_distance += haversine(
                self.lat[p1], self.lon[p1],
                self.lat[p2], self.lon[p2],
            )
        return total_distance

    def score(self):
        """
        (1) Distance Matrix is calculated using haversine distance
        (2) Graph and index graph are calculated
        (3) Based on the maximum reachable distance in the last layer, the
            index Graph is traversed to find the corresponding indices
        """
        if not(len(self.alt) == len(self.lat) == len(self.lon)):
            return []

        latlon = np.transpose(np.vstack([np.radians(self.lat), np.radians(self.lon)]))
        dist_matrix = self.simple_dist_matrix(latlon)
        graph, index_graph = self.find_graph(dist_matrix)

        return self.find_path(index_graph, reverse_from=np.argmax(graph[:,self.layers-1]))

    def score_with_height(self, epsilon=None):
        if not(len(self.alt) == len(self.lat) == len(self.lon)):
            return []

        def check_alt(alt, path):
            return alt[path[0]]-alt[path[-1]] <= 1000

        latlon = np.column_stack([np.radians(self.lat), np.radians(self.lon)])

        dist_matrix = self.simple_dist_matrix(latlon)
        graph, index_graph = self.find_graph(dist_matrix)
        path = self.find_path(index_graph, reverse_from=np.argmax(graph[:,self.layers-1]))

        if check_alt(self.alt, path):
            return path

        calculated = []
        lower_bound = 0
        lower_bound_km = 0
        best_path = []

        while True:
            reverse_from = np.argmax(graph[:,self.layers-1])
            forbidden_start_index = np.nonzero(self.alt-self.alt[reverse_from] > 1000)[0]

            fake_dist_matrix = np.copy(dist_matrix)
            fake_dist_matrix[forbidden_start_index,:] = -10000

            # only use the allowed indexes for the first turnpoint
            height_graph, height_index_graph = self.find_graph(dist_matrix, fake_dist_matrix)
            path = self.find_path(height_index_graph, reverse_from=reverse_from)

            calculated.append(path[-1])
            height_graph[calculated, self.layers-1] = 0
            graph[calculated,self.layers-1] = 0

            # some tests while developing
            for j in forbidden_start_index:
                assert(path[0]!=j)
            assert(check_alt(self.alt, path)) # careful with self.alt alt

            distance = self.distance_from_graph(height_graph)
            if distance > lower_bound:
                lower_bound = distance
                lower_bound_km = self.find_distance(path)
                best_path = path

            print(f'Lower bound: {lower_bound_km}')

            # do we still have options to check?
            remaining = np.nonzero(graph[:,self.layers-1] > lower_bound)[0]
            if not len(remaining):
                return best_path
