import math
import time
import aerofiles.util.pressure as pr
import numpy as np
import datetime as dt

from .config import FlightParsingConfig
from  aerofiles.util.viterbi import SimpleViterbiDecoder
from aerofiles.util.geo import haversine_distance, EARTH_RADIUS_KM


class EmissionGenerator:
    """Compute stuff on already saved flight and fix instances"""
    def __init__(self, data, sensor_type=None, config_class=FlightParsingConfig):
        self.len = len(data['lon'])

        self.lon = data['lon']
        self.lat = data['lat']
        self.alt = data['alt']
        self.time = data['time']
        self.raw_time = data['raw_time']
        self.raw_time_diff = data['raw_time_diff']
        self.sensor = data.get('sensor')
        self.sensor_type = sensor_type

        self.takeoff = None
        self.landing = None
        self.valid = True

        self.config = config_class()
        self.notes = []

    def run(self):
        self.compute_speeds()
        self.compute_flight()
        self.compute_takeoff_landing()

        self.compute_bearings()
        self.compute_bearing_change_rates()

        if self.sensor_type is not None:
            self.compute_engine()

        self.compute_tow()
        self.compute_windows()
        self.compute_circling()
        self.compute_thermals()

        return {
            'distance': self.h_dist,
            'track_distance': self.track_dist,
            'ground_speed': self.ground_speed,
            'vario': self.vario,
            'bearing': self.bearing,
            'bearing_change_rate': self.bearing_change_rate,
            'tow': self.tow,
            'windows': self.windows,
            'glides': self.glides,
            'thermals': self.thermals,
            'notes': self.notes
        }

    def compute_speeds(self):
        """Adds horizontal speed (km/h) and vertical speed (m/s) to self.fixes."""
        lat = np.radians(self.lat)
        lon = np.radians(self.lon) * np.cos(lat)

        lon_diff = np.diff(lon, prepend=lon[0])
        lat_diff = np.diff(lat, prepend=lat[0])

        self.h_dist = np.hypot(lon_diff, lat_diff) * EARTH_RADIUS_KM
        # Track distances stores distance accumulated while in glide mode
        self.v_dist = np.diff(self.alt, prepend=self.alt[0])

        assert(np.shape(self.h_dist)==np.shape(self.v_dist))
        self.ground_speed = np.divide(
            self.h_dist*3600,
            self.raw_time_diff,
            out=np.zeros_like(self.h_dist, dtype=np.float64),
            where=self.raw_time_diff!=0,
        )
        self.vario = np.divide(
            self.v_dist,
            self.raw_time_diff,
            out=np.zeros_like(self.v_dist, dtype=np.float64),
            where=self.raw_time_diff!=0,
        )

    def compute_flight(self):
        """
        Two pass:
          1. Viterbi decoder
          2. Only emit landings (0) if the downtime is more than
             _config.min_landing_time (or it's the end of the log).
        """
        emissions = (self.ground_speed > self.config.min_gsp_flight)
        emissions = emissions.tolist()
        decoder = SimpleViterbiDecoder(
            # More likely to start the log standing, i.e. not in flight
            init_probs=[0.80, 0.20],
            transition_probs=[
                [0.9995, 0.0005],  # transitions from standing
                [0.0005, 0.9995],  # transitions from flying
            ],
            emission_probs=[
                [0.8, 0.2],  # emissions from standing
                [0.2, 0.8],  # emissions from flying
            ])

        outputs = decoder.decode(emissions)
        # Step 2: apply _config.min_landing_time.
        ignore_next_downtime = False
        apply_next_downtime = True # originally False
        self.flying = np.zeros_like(outputs)
        for i, output in enumerate(outputs):
            if output == 1:
                self.flying[i] = True
                # We're in flying mode, therefore reset all expectations
                # about what's happening in the next down mode.
                ignore_next_downtime = False
                apply_next_downtime = False
            else:
                if apply_next_downtime or ignore_next_downtime:
                    if apply_next_downtime:
                        self.flying[i] = False
                    else:
                        self.flying[i] = True
                else:
                    # We need to determine whether to apply_next_downtime
                    # or to ignore_next_downtime. This requires a scan into
                    # upcoming fixes. Find the next fix on which
                    # the Viterbi decoder said "flying".
                    j = i + 1
                    while j < self.len:
                        upcoming_fix_decoded = outputs[j]
                        if upcoming_fix_decoded == 1:
                            break
                        j += 1

                    if j == self.len:
                        # No such fix, end of log. Then apply.
                        apply_next_downtime = True
                        self.flying[i] = False
                    else:
                        # Found next flying fix.
                        upcoming_fix = self.fixes[j]
                        upcoming_fix_time_ahead = self.raw_time[j] - self.raw_time[i]
                        # If it's far enough into the future of then apply.
                        if upcoming_fix_time_ahead >= self.config.min_landing_time:
                            apply_next_downtime = True
                            self.flying[i] = False
                        else:
                            ignore_next_downtime = True
                            self.flying[i] = True

    def compute_takeoff_landing(self):
        """Finds the takeoff and landing fixes in the log.
        Takeoff fix is the first fix in the flying mode. Landing fix
        is the next fix after the last fix in the flying mode or the
        last fix in the file.
        """
        was_flying = False
        for i in range(self.len):
            if self.flying[i] and self.takeoff is None:
                self.takeoff = i
            if not self.flying[i] and was_flying:
                self.landing = i
                if self.config.which_flight_to_pick == "first":
                    # User requested to select just the first flight in the log,
                    # terminate now.
                    break
            was_flying = self.flying[i]

        if not self.takeoff:
            self.notes.append('Error: Did not detect takeoff.')
            self.valid = False

        if not self.landing:
            self.notes.append('Error: Did not detect landing.')

    def compute_engine(self):
        """
        Two pass:
          1. Viterbi decoder
          2. Only emit engine on (1) if the downtime is more than
             _config.min_engine_running.
        """

        # Step 1: the Viterbi decoder
        emissions = self.engine_emissions().tolist()
        decoder = SimpleViterbiDecoder(
            # More likely to start with engine on
            init_probs=[0.40, 0.60],
            transition_probs=[
                [0.9995, 0.0005],  # transitions from engine off
                [0.0005, 0.9995],  # transitions from engine on
            ],
            emission_probs=[
                [0.8, 0.2],  # emissions from engine off
                [0.2, 0.8],  # emissions from engine on
            ])

        outputs = decoder.decode(emissions)
        # TODO: Add more logic here, soft minima can be applied.
        self.engine = outputs

    def compute_tow(self):
        """Computes boolean array tow.

        This is independend of engine. Even if the engine is running, the
        glider might be on tow (push-pull). Therefore, tow must be False
        for self-launches to start scoring
        """
        self.tow = np.zeros(self.len, dtype=bool)

        emissions = self.tow_emissions().tolist()
        # emissions[0] = 1 # start on tow
        decoder = SimpleViterbiDecoder(
            # Flight is most likely to start with tow
            # Value set to lower again, as takeoff is not easy to
            # regard as tow
            init_probs=[0.1, 0.9],
            transition_probs=[
                [0.999999999, 0.000000001],  # transitions normal flight to tow
                [0.01, 0.99],  # transitions from tow to normal flight
            ],
            emission_probs=[
                [0.5, 0.5],  # emissions from off tow
                [0.1, 0.9],  # emissions from on tow
            ])

        start = self.takeoff or 0
        tow_started = False
        outputs = decoder.decode(emissions[start:])

        for i in range(self.len-start):
            tow = outputs[i]
            self.tow[start+i] = tow
            if tow and not tow_started:
                tow_started = True
                # if we detect a tow
                self.tow[start:start+i] = 1
            if not tow and tow_started:
                return

    def compute_windows(self):
        """Compute all possible scoring windows of the flight.
        Windows stores rows of scoring_windows [start_index, end_index]
        where start_index is the first fix from the scoring intervall and
        end_fix is the first fix which does not belong to the intervall anymore.
        """
        self.scoring = np.logical_and(
            np.logical_not(self.tow),
            self.flying,
        )
        if self.sensor_type is not None:
            self.scoring = np.logical_and(
                self.scoring,
                self.flying,
                np.logical_not(self.engine),
            )

        windows = np.flatnonzero(np.insert(np.diff(self.scoring), 0, self.scoring[0]))
        # if windows has odd length, then last index is end of last scoring intervall
        if len(windows) % 2:
            windows = np.append(windows, self.len)
        self.windows = windows.reshape(-1, 2)

    def compute_bearings(self):
        """Computes bearing"""
        lat = np.radians(self.lat)
        lon = np.radians(self.lon)
        lon_diff = np.diff(lon)

        y = np.sin(lon_diff) * np.cos(lat[1:])
        x = (np.cos(lat[:-1]) * np.sin(lat[1:]) -
             np.sin(lat[:-1]) * np.cos(lat[1:]) * np.cos(lon_diff))
        bearing = np.degrees(np.arctan2(y, x))
        self.bearing = np.insert(bearing, 0, bearing[0])

    def compute_bearing_change_rates(self):
        """
        Computing bearing change rate between neighboring fixes proved
        itself to be noisy on tracks recorded with minimum interval (1 second).
        Therefore we compute rates between points that are at least
        min_time_for_bearing_change seconds apart.
        """
        def find_prev_fix(i):
            """Computes the previous fix to be used in bearing rate change."""
            prev_fix = None
            for j in range(i-1, 0, -1):
                if (np.abs(self.raw_time[i] - self.raw_time[j]) >
                        self.config.min_time_for_bearing_change - 1e-7):
                    prev_fix = j
                    break
            return prev_fix

        self.bearing_change_rate = np.zeros(self.len)
        for curr_fix in range(self.len):
            prev_fix = find_prev_fix(curr_fix)
            if prev_fix is not None:
                bearing_change = self.bearing[prev_fix] - self.bearing[curr_fix]
                if np.abs(bearing_change) > 180.0:
                    if bearing_change < 0.0:
                        bearing_change += 360.0
                    else:
                        bearing_change -= 360.0
                time_change = self.raw_time[prev_fix] - self.raw_time[curr_fix]
                self.bearing_change_rate[curr_fix] = bearing_change/time_change

    def circling_emissions(self):
        """Generates raw circling/straight emissions from bearing change.
        Staight flight is encoded as 0, circling is encoded as 1. Exported
        to a separate function to be used in Baum-Welch parameters learning.
        """
        return np.logical_and(
            self.flying,
            (np.absolute(self.bearing_change_rate) >
            self.config.min_bearing_change_circling)
        )

    def engine_emissions(self):
        """Generates raw engine/no-engine emissions from engine sensor.
        Which sensor to pick is defined in parsing_config
        engine off is encoded as 0, engine running is encoded as 1. Exported
        to a separate function to be used in Baum-Welch parameters learning.
        """
        return self.sensor > self.config.min_sensor_level[self.sensor_type]

    def tow_emissions(self):
        """Generates raw on-tow/off-tow emissions from bearing change rate
        and vario. Exported to a separate function to be used
        in Baum-Welch parameters learning.
        """
        return np.logical_and(
            np.absolute(self.bearing_change_rate) < self.config.max_bearing_change_on_tow,
            self.vario > self.config.min_vario_on_tow
        )

    def compute_circling(self):
        """Computes circling array"""
        emissions = self.circling_emissions().tolist()
        decoder = SimpleViterbiDecoder(
            # More likely to start in straight flight than in circling
            init_probs=[0.80, 0.20],
            transition_probs=[
                [0.982, 0.018],  # transitions from straight flight
                [0.030, 0.970],  # transitions from circling
            ],
            emission_probs=[
                [0.942, 0.058],  # emissions from straight flight
                [0.093, 0.907],  # emissions from circling
            ])

        self.circling = np.array(decoder.decode(emissions),dtype=bool)
        # Now we can calculate track distance
        self.track_dist = np.where(~self.circling, self.h_dist, 0)

    def compute_thermals(self):
        """
        Go through the fixes and find the thermals.
        Every point not in a thermal is put into a glide. If we get to end of
        the fixes and there is still an open glide (i.e. flight not finishing
        in a valid thermal) the glide will be closed.
        """
        thermals = []
        glides = []
        if not self.takeoff:
            return

        if not self.landing:
            landing_index = self.fixes[-1].index

        circling_now = False
        gliding_now = False
        first = None
        first_glide = None
        last_glide = None
        for i in range(self.len):
            circling = self.circling[i]
            if not circling_now and circling:
                # Just started circling
                circling_now = True
                first = i
            elif circling_now and not circling:
                # Just ended circling
                circling_now = False

                time_change = self.raw_time[i] - self.raw_time[first]
                if time_change > self.config.min_time_for_thermal - 1e-5:
                    thermals.append([first, i])
                    # glide ends at start of thermal
                    glides.append([first_glide, first])
                    gliding_now = False

            if gliding_now:
                last_glide = i
            else:
                # just started gliding
                first_glide = i
                last_glide = i
                gliding_now = True

        if gliding_now:
            glides.append([first_glide, last_glide])

        self.glides = np.array(glides)
        self.thermals = np.array(thermals)

        for glide in self.glides[:-1]:
            assert(glide[1] in self.thermals[:,0])
        for thermal in self.thermals[:-1]:
            assert(thermal[1] in self.glides[:,0])
