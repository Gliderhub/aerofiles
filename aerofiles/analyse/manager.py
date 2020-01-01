import os
import numpy as np
import datetime as dt

from .score import Scorer as OLCScorer
from .stats import Analyser as Analyser
from .emissions import EmissionGenerator
from aerofiles.igc import Reader
from aerofiles.analyse.config import FlightParsingConfig as Config

class FlightManager:
    def __init__(self):
        self.test_data_dir = 'aerofiles/analyse/test_data'
        self.test_file_names = {
            'torben': '87ilqqk1.igc',
            'sebald1': '88qd4er1.igc',
            'moritz_wave': '825lqkk1.igc',
            'simple_thermal_moritz': '75lv20o1.igc'
            # https://www.onlinecontest.org/olc-3.0/gliding/flightinfo.html?dsId=5722478
            }
        self.data = {}
        self.notes = []
        self.sensor = None

    def import_flight(self, file_name):
        file_name = self.test_file_names[file_name]
        test_file = os.path.join(self.test_data_dir, file_name)
        with open(test_file, 'r') as f:
            self.parsed = Reader().read(f)

    def set_up(self, sensor_required=False):
        """
        This method sets up the data as it will get passed in the pipeline later
        on
        """
        records = self.parsed['fix_records'][1]
        available_ext = [
            ext['extension_type'] for ext in self.parsed['fix_record_extensions'][1]
        ]
        sensors = [s for s in Config.sensors if s in available_ext]

        if sensor_required:
            if sensors:
                self.data['sensor'] = np.array([r[sensors[0]] for r in records])
                self.sensor = sensors[0]
            else:
                self.notes.append('No engine sensor found')

        utc_date = self.parsed['header'][1]['utc_date']

        self.data['lat'] = np.array([r['lat'] for r in records])
        self.data['lon'] = np.array([r['lon'] for r in records])
        self.data['time'] = np.array(
            [dt.datetime.combine(utc_date, r['time']) for r in records]
        )
        self.data['raw_time'] = np.array([
            ((r['time'].hour * 60)
            + r['time'].minute) * 60
            + r['time'].second for r in records
        ])
        self.data['raw_time_diff'] = np.diff(
            self.data['raw_time'],
            prepend=[self.data['raw_time'][0]]
        )
        self.data['alt'] = np.array([r['pressure_alt'] for r in records])

    def run_contest(self):
        """
        Support for different contest types will be added, emissions should
        then be moved outside this method, as they only need to be calculated
        once
        """
        generator = EmissionGenerator(self.data, self.sensor)
        self.data.update(generator.run())

        # TODO: Add options for multiple scoring windows
        # We find the best scoring window and use it for scoring
        best_distance = 0
        best_path = []
        if len(self.data['windows']):
            for start, end in self.data['windows']:
                # Debugging print
                start_time = self.data['time'][start]
                end_time = self.data['time'][end]
                print(f'Window found: From {start_time} to {end_time}')

                scorer = OLCScorer(self.data, start, end)
                path = scorer.score_with_height_backwards()
                distance = scorer.find_distance(path)
                if distance >= best_distance:
                    best_distance = distance
                    best_path = path + start # start index needs to be added
        else:
            self.notes.append('No scoring windows found')
            return
        analyser = Analyser()
        contest, legs = analyser.analyse(self.data, best_path)
