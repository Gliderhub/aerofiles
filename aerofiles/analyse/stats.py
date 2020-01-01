from aerofiles.util.geo import haversine_distance, track_distance


class Analyser:
    def __init__(self):
        pass

    def analyse(self, lat, lon, alt, time, raw_time, path):
        self.legs = []
        self.contest = {}

        for i in range(len(path)-1):
            leg = {}
            leg['start'] = path[i]
            leg['end'] = path[i+1]

            leg['start_time'] = time[path[i]]
            leg['end_time'] = time[path[i+1]]

            leg['start_alt'] = alt[path[i]]
            leg['end_alt'] = alt[path[i+1]]

            leg['start_point'] = (lon[path[i]], lat[path[i]])
            leg['end_point'] = (lon[path[i+1]], lat[path[i+1]])

            leg['distance'] = haversine_distance(
                lon[path[i]], lat[path[i]],
                lon[path[i+1]], lat[path[i+1]],
            )
            leg['speed'] = (
                3600 * leg['distance'] /
                (time[path[i+1]]-time[path[i]]).total_seconds()
            )
            leg['track_distance'] = track_distance(
                lon[path[i]:path[i+1]],
                lat[path[i]:path[i+1]],
                )
            leg['glide_ratio'] = (
                leg['track_distance'] /
                alt[path[i+1]] - alt[path[i]]
            )
            self.legs.append(leg)

        self.contest['distance'] = sum([leg['distance'] for leg in self.legs])
        self.contest['track_distance'] = sum(
            [leg['track_distance'] for leg in self.legs]
        )
        self.contest['start_time'] = time[path[0]]
        self.contest['end_time'] = time[path[-1]]
        self.contest['speed'] = (
            3600*self.contest['distance'] /
            (time[path[-1]]-time[path[0]]).total_seconds()
        )

        return self.contest, self.legs


    def compute_circling():
        pass

    def find_thermals():
        pass

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
