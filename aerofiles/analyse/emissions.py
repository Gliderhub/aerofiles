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
    def __init__(self, lon, lat, alt, time, raw_time, sensor, config_class=FlightParsingConfig):
        self.len = len(lon)

        self.lon = np.array(lon)
        self.lat = np.array(lat)
        self.alt = np.array(alt)
        self.time = np.array(time)
        self.raw_time = np.array(raw_time)
        self.raw_time_diff = np.diff(self.raw_time, prepend=[self.raw_time[0]])
        self.sensor = np.array(sensor)

        self.takeoff = None
        self.landing = None
        self.valid = True

        self.sensor_type = 'ENL'

        self.config = config_class()
        self.notes = []

    def run(self):

        self.compute_speeds()
        self.compute_flight()
        self.compute_takeoff_landing()
        self.compute_engine()
        self.compute_bearings()
        self.compute_bearing_change_rates()
        # self.compute_tow()
        # self.compute_windows()
        return 0
        # return {
        #     'distances': self.distances,
        #     'glides': self.glides,
        #     'thermals': self.thermals,
        #     'windows': self.windows,
        #     'ground_speed': self.ground_speed,
        #     'vario': self.vario,
        #     'bearings': self.bearings,
        #     'tow': self.tow,
        #     'bearing_change': self.bearing_change_rate,
        #     'notes': self.notes
        # }

    def compute_speeds(self):
        """Adds horizontal speed (km/h) and vertical speed (m/s) to self.fixes."""
        lat = np.radians(self.lat)
        lon = np.radians(self.lon) * np.cos(lat)

        lon_diff = np.diff(lon, prepend=lon[0])
        lat_diff = np.diff(lat, prepend=lat[0])

        self.h_dist = np.hypot(lon_diff, lat_diff) * EARTH_RADIUS_KM
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
        print(self.ground_speed)
        emissions = (self.ground_speed > self.config.min_gsp_flight)
        # print(emissions)
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
        emissions = self.engine_emissions(self.sensor).tolist()
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
        """Adds boolean flag .tow to self.fixes.

        This is independend of .engine. Even if the engine is running, the
        glider might be on tow (push-pull). Therefore, no .tow must False
        for self-launches to start scoring
        """
        emissions = self.tow_emissions()
        emissions[0] = 1 # start on tow
        decoder = SimpleViterbiDecoder(
            # Flight is most likely to start with tow
            init_probs=[0.00000001, 0.99999999],
            transition_probs=[
                [0.999999999, 0.000000001],  # transitions normal flight to tow
                [0.01, 0.99],  # transitions from tow to normal flight
            ],
            emission_probs=[
                [0.5, 0.5],  # emissions from off tow
                [0.1, 0.9],  # emissions from on tow
            ])

        tow_started = False
        tow_ended = False
        outputs = decoder.decode(emissions[self.takeoff:])
        # TODO: Add more logic here, soft minima can be applied.
        for i in range(self.len):
            if i < self.takeoff:
                tow[i] = False
            elif tow_ended:
                tow[i] = False
            else:
                tow = (outputs[i-self.takeoff] == 1)
                if tow and not tow_started:
                    # TOTHINK: Do we need to save the tow start/end fix?
                    # self.flight.start_tow_fix = self.fixes[i]
                    tow_started = True
                if tow_started and not tow:
                    tow_ended = True
                    # self.end_tow_fix = self.fixes[i]
                tow[i] = tow

    def compute_windows(self):
        """Compute all possible scoring windows of the flight.
        started_fix is the first fix of the window and ended_fix is the first fix
        after the window has ended.
        """
        window_started = False
        for i in range(self.len):
            if scoring[i] and not window_started:
                window_started = True
                start = i
            if not scoring[i] and window_started:
                window_started = False
                end = i
                self.windows.append((start, end))
        # window still open after iterating all fixes
        if window_started:
            end = self.len-1
            self.windows.append((start, end))

    def compute_bearings(self):
        """Computes bearing"""
        lat = np.radians(self.lat)
        lon = np.radians(self.lon)
        lon_diff = np.diff(lon)

        y = np.sin(lon_diff) * np.cos(lat[1:])
        x = (np.cos(lat[:-1]) * np.sin(lat[1:]) -
             np.sin(lat[:-1]) * np.cos(lat[1:]) * np.cos(lon_diff))
        bearing = np.degrees(np.arctan2(y, x))


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

        bearing_change_rate = np.zeros(self.len)
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
                bearing_change_rate[curr_fix] = bearing_change/time_change

    def circling_emissions(self):
        """Generates raw circling/straight emissions from bearing change.
        Staight flight is encoded as 0, circling is encoded as 1. Exported
        to a separate function to be used in Baum-Welch parameters learning.
        """
        return np.logical_and(
            flying,
            np.absolute(bearing_change_rate) > self.config.min_bearing_change_circling
        )

    def engine_emissions(self, sensor):
        """Generates raw engine/no-engine emissions from engine sensor.
        Which sensor to pick is defined in parsing_config
        engine off is encoded as 0, engine running is encoded as 1. Exported
        to a separate function to be used in Baum-Welch parameters learning.
        """
        return sensor > self.config.min_sensor_level[self.sensor_type]

    def tow_emissions(self):
        """Generates raw on-tow/off-tow emissions from bearing change rate
        and vario. Exported to a separate function to be used
        in Baum-Welch parameters learning.
        """
        return np.logical_and(
            np.absolute(bearing_change_rate) < self.config.max_bearing_change_on_tow,
            vario > self.config.min_vario_on_tow
        )

    def compute_circling(self):
        """Adds .circling to self.fixes."""
        emissions = self.circling_emissions()
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

        output = decoder.decode(emissions)

    def find_thermals(self):
        """
        Go through the fixes and find the thermals.
        Every point not in a thermal is put into a glide. If we get to end of
        the fixes and there is still an open glide (i.e. flight not finishing
        in a valid thermal) the glide will be closed.
        """
        if not self.takeoff:
            return

        if not self.landing:
            landing_index = self.fixes[-1].index

        circling_now = False
        gliding_now = False
        first_fix = None
        first_glide_fix = None
        last_glide_fix = None
        distance = 0.0
        for fix in flight_fixes:
            if not circling_now and fix.circling:
                # Just started circling
                circling_now = True
                first_fix = fix
                distance_start_circling = distance
            elif circling_now and not fix.circling:
                # Just ended circling
                circling_now = False
                first_fix.flight = self.flight
                first_fix.save()
                fix.flight = self.flight
                fix.save()
                thermal = Thermal(
                    start_fix=first_fix,
                    end_fix=fix
                )
                self.thermals.append(thermal)
                if (thermal.time_change >
                        self.config.min_time_for_thermal - 1e-5):
                    self.thermals.append(thermal)
                    first_glide_fix.flight = self.flight
                    first_glide_fix.save()
                    first_fix.flight = self.flight
                    first_fix.save()
                    # glide ends at start of thermal
                    glide = Glide(
                        start_fix=first_glide_fix,
                        end_fix=first_fix,
                        track_length=distance_start_circling
                    )
                    self.glides.append(glide)
                    gliding_now = False

            if gliding_now:
                distance = distance + fix.distance_to(last_glide_fix)
                last_glide_fix = fix
            else:
                # just started gliding
                first_glide_fix = fix
                last_glide_fix = fix
                gliding_now = True
                distance = 0.0

        if gliding_now:
            first_glide_fix.flight = self.flight
            first_glide_fix.save()
            last_glide_fix.flight = self.flight
            last_glide_fix.save()
            glide = Glide(
                start_fix=first_glide_fix,
                end_fix=last_glide_fix,
                track_length=distance
            )
            self.glides.append(glide)

        for glide in self.glides:
            glide.flight = self.flight
            glide.save()
        for thermal in self.thermals:
            thermal.flight = self.flight
            thermal.save()
