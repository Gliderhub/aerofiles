import os
import numpy as np
import scipy
import datetime
from sklearn.neighbors import DistanceMetric
from math import radians

from igc import Reader


class Scorer:
    """
    Find polygonal line of maximal length with data points as vertices.

    Height difference between starting point and end point is maximal 1000m.
    """

    def __init__(self):
        self.layers = 6
        self.test_data_dir = 'score/test_data'

    def import_perlan(self):
        TOW_RELEASE_PERLAN = datetime.time(16, 54, 10)

        test_file = os.path.join(self.test_data_dir, '99bv7r92.igc')
        with open(test_file, 'r') as f:
            parsed = Reader().read(f)
        records = parsed['fix_records'][1]
        for i, record in enumerate(records):
            if record['time'] > TOW_RELEASE_PERLAN:
                tow_release_index = i
                break

        records = records[tow_release_index:]
        self.lat = np.array([r['lat'] for r in records])
        self.lon = np.array([r['lon'] for r in records])
        self.alt = np.array([r['pressure_alt'] for r in records])

    def import_moflight(self):
        test_file = os.path.join(self.test_data_dir, '825lqkk1.igc')
        with open(test_file, 'r') as f:
            parsed = Reader().read(f)

        records = parsed['fix_records'][1]
        self.lat = np.array([r['lat'] for r in records])
        self.lon = np.array([r['lon'] for r in records])
        self.alt = np.array([r['pressure_alt'] for r in records])

    def sklearn_haversine(self, latlon):
        haversine = DistanceMetric.get_metric('haversine')
        dists = haversine.pairwise(latlon)
        return 6371 * dists


    def find_graph(self, real_dist_matrix, fake_dist_matrix=None):
        """
        Calculates (k,l) shaped graph where k is the number of knots
        (data points) and l is the number of layers or legs.

        Graph is used to store the optimum distance that can be achieved with
        l layers and knot k.

        The index graph stores the indices of the previous knot.

        Fake_dist_matrix is used to only allow certain start indices that obey
        the height constrain.
        """
        knots = np.shape(real_dist_matrix)[0]

        graph = np.zeros((knots,self.layers))
        index_graph = np.zeros((knots,self.layers), dtype='int32')

        if fake_dist_matrix is not None:
            dist_matrix = fake_dist_matrix
        else:
            dist_matrix = real_dist_matrix
        
        for k in range(0, knots):
            index_graph[k,0] = np.argmax(dist_matrix[:k+1,k])
            graph[k,0] = np.max(dist_matrix[:k+1,k])

        for k in range(0, knots):
            for l in range(1, self.layers):
                index_graph[k,l] = np.argmax(graph[:k+1,l-1] + real_dist_matrix[:k+1,k])
                graph[k,l] = np.max(graph[:k+1,l-1] + real_dist_matrix[:k+1,k])

        return graph, index_graph


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

    def find_distance(self, graph):
        return np.max(graph[:,self.layers-1])

    def score(self):
        """
        (1) Distance Matrix is calculated using haversine distance

        (2) Graph and index graph are calculated

        (3) Based on the maximum reachable distance in the last layer, the
            index Graph is traversed to find the corresponding indices
        """

        latlon = np.transpose(np.vstack([np.radians(self.lat), np.radians(self.lon)]))
        dist_matrix = self.sklearn_haversine(latlon)
        graph, index_graph = self.find_graph(dist_matrix)

        return self.find_path(index_graph, reverse_from=np.argmax(graph[:,self.layers-1]))

    def check_alt(self, path):
        return self.alt[path[0]]-self.alt[path[-1]] <= 1000

    def score_with_height(self):
        latlon = np.transpose(np.vstack([np.radians(self.lat), np.radians(self.lon)]))
        dist_matrix = self.sklearn_haversine(latlon)

        graph, index_graph = self.find_graph(dist_matrix)
        path = self.find_path(index_graph, reverse_from=np.argmax(graph[:,self.layers-1]))

        # Height constrain already fullfilled
        if self.check_alt(path):
            return path

        calculated = []
        lower_bound = 0
        while True:
            print(f'Iteration: {iteration}, Lower bound: {lower_bound}')
            reverse_from = np.argmax(graph[:,layers-1])
            forbidden_start_index = np.nonzero(self.alt-self.alt[reverse_from] > 1000)[0]

            fake_dist_matrix = np.copy(dist_matrix)
            fake_dist_matrix[forbidden_start_index,:] = -10000

            # only use the allowed indexes for the first turnpoint
            height_graph, height_index_graph = self.find_graph(dist_matrix, fake_dist_matrix)
            path = self.find_path(height_index_graph, reverse_from=reverse_from)

            calculated.append(path[-1])
            height_graph[calculated, self.layers-1] = 0
            graph[calculated,self.layers-1] = 0

            for j in forbidden_start_index:
                assert(path[0]!=j)

            if not self.check_alt(path):
                print(self.alt[path[0]], self.alt[path[-1]])
                return

            distance = self.find_distance(height_graph)
            if distance > lower_bound:
                lower_bound = distance
                best_path = path

            # do we still have options to check?
            remaining = np.nonzero(graph[:,layers-1] > lower_bound)[0]
            if not len(remaining):
                return best_path


if __name__ == '__main__':

    scorer = Scorer()
    scorer.import_moflight()
    path = scorer.score_with_height()
    print(path)
    print(np.array(scorer.alt)[path])
