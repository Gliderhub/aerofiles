import numpy as np
from aerofiles.util.geo import haversine


class Analyser:
    def __init__(self):
        pass
    # TODO: Add zero division handling everywhere

    def analyse(self, data, path):
        self.legs = []
        self.contest = {}

        # Do we really need to do this?
        time = data['time']
        lon = data['lon']
        lat = data['lat']
        time = data['time']
        alt = data['alt']
        thermals = data['thermals']
        glides = data['glides']
        raw_time = data['raw_time']
        distance = data['distance']
        track_distance = data['track_distance']
        # cumulative distance is calculated over the whole path, not only
        # from scoring start, not a problem because all calculation involve
        # subtraction of cumulative sums, so it equals off
        cum_track_distance = np.cumsum(track_distance)

        # iterate all legs
        for start, stop in zip(path, path[1:]):
            leg = {}
            leg['raw_time'] = raw_time[stop] - raw_time[start]
            # only thermals fully or partly in the current leg are considered
            # and clipped of at the leg start
            leg_thermals = thermals[(thermals[:,1]>start) & (thermals[:,0]<stop)]
            leg_thermals = np.clip(leg_thermals, start, stop)

            leg['thermal_count'] = len(leg_thermals)
            leg['thermal_gain'] = np.sum(
                alt[leg_thermals[:,1]]-alt[leg_thermals[:,0]]
            )
            leg['thermal_time'] = np.sum(
                raw_time[leg_thermals[:,1]]-raw_time[leg_thermals[:,0]]
            )
            leg['thermal_avg'] = leg['thermal_gain'] / leg['thermal_time']
            leg['thermal_percentage'] = leg['thermal_time'] / leg['raw_time']

            # only glides fully or partly in the current leg are considered
            # and clipped of at the leg start
            leg_glides = glides[(glides[:,1]>start) & (glides[:,0]<stop)]
            leg_glides = np.clip(leg_glides, start, stop)

            leg['glide_count'] = len(leg_glides)
            leg['glide_gain'] = np.sum(
                alt[leg_glides[:,1]] -
                alt[leg_glides[:,0]]
            )
            leg['glide_time'] = np.sum(
                raw_time[leg_glides[:,1]] -
                raw_time[leg_glides[:,0]]
            )

            # glide_track is in KM
            leg['glide_distance'] = np.sum(
                cum_track_distance[leg_glides[:,1]] -
                cum_track_distance[leg_glides[:,0]]
            )
            # per definition of L/D, we need the negative sign
            leg['glide_ratio'] = -(leg['glide_track']/leg['glide_gain']) * 1000
            leg['glide_percentage'] = leg['glide_time']/leg['raw_time']

            assert(leg['glide_time']+leg['thermal_time']==leg['raw_time'])
            assert(leg['glide_gain']+leg['thermal_gain']==alt[stop]-alt[start])

            leg['start'] = start
            leg['end'] = stop

            leg['start_time'] = time[start]
            leg['end_time'] = time[stop]

            leg['start_alt'] = alt[start]
            leg['end_alt'] = alt[stop]

            leg['start_point'] = (lon[start], lat[start])
            leg['end_point'] = (lon[stop], lat[stop])

            leg['distance'] = haversine(
                lon[start], lat[start],
                lon[stop], lat[stop],
            )
            leg['speed'] = 3600 * leg['distance'] / leg['raw_time']
            self.legs.append(leg)

        self.contest['distance'] = sum([leg['distance'] for leg in self.legs])
        self.contest['glide_distance'] = sum(
            [leg['glide_track'] for leg in self.legs]
        )
        self.contest['start_time'] = time[path[0]]
        self.contest['end_time'] = time[path[-1]]

        self.contest['raw_time'] = raw_time[path[-1]] - raw_time[path[0]]
        self.contest['speed'] = (
            3600*self.contest['distance'] /
            self.contest['raw_time']
        )
        # We can not sum up over the legs because clipped glides would count twice
        self.contest['glide_count'] = len(
            glides[(glides[:,1]>path[0]) & (glides[:,0]<path[-1])]
        )
        self.contest['glide_gain'] = sum(
            [leg['glide_gain'] for leg in self.legs]
        )
        self.contest['glide_time'] = sum(
            [leg['glide_time'] for leg in self.legs]
        )
        self.contest['glide_track'] = sum(
            [leg['glide_track'] for leg in self.legs]
        )
        self.contest['glide_ratio'] = -(
            self.contest['glide_track'] / self.contest['glide_gain']
        ) * 1000
        self.contest['glide_percentage'] = (
            self.contest['glide_time'] / self.contest['raw_time']
        )
        # We can not sum up over the legs because clipped thermals would count twice
        self.contest['thermal_count'] = len(
            thermals[(thermals[:,1]>path[0]) & (thermals[:,0]<path[-1])]
        )
        self.contest['thermal_gain'] = sum(
            [leg['thermal_gain'] for leg in self.legs]
        )
        self.contest['thermal_time'] = sum(
            [leg['thermal_time'] for leg in self.legs]
        )
        self.contest['thermal_avg'] = (
            self.contest['thermal_gain'] / self.contest['thermal_time']
        )
        self.contest['thermal_percentage'] = (
            self.contest['thermal_time'] / self.contest['raw_time']
        )

        return self.contest, self.legs


    def compute_qnh(self):
        if not (self.fixes and self.flight.takeoff_fix):
            # TODO: AGL information can be used here to infere qnh
            self.flight.qnh = None
            return
        # TODO: Which fixes to take for qnh guessing
        # TOTHINK: More logic needed here, maybe an exta class
        sum = 0
        for fix in self.fixes[0:self.flight.takeoff_fix.index+1]:
            known_alt = fix.ground_alt if fix.ground_alt else fix.gnss_alt
            sum += pr.find_qnh(fix.pressure_alt, known_alt)

        qnh = round(sum / len(self.fixes[0:self.flight.takeoff_fix.index+1]))
        self.flight.qnh = qnh
        for fix in self.fixes:
            fix.qnh_alt = pr.pressure_alt_to_qnh_alt(fix.pressure_alt, qnh)
