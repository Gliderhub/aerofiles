from aerofiles.score import Scorer as OLCScorer
from .stats import Analyser as Analyser
from .emissions import EmissionGenerator

def test_pipeline():


    scorer = OLCScorer()
    scorer.import_torben_flight()
    lon, lat, alt = list(scorer.lon), list(scorer.lat), list(scorer.alt)
    time, raw_time, sensor = list(scorer.time), list(scorer.raw_time), list(scorer.sensor)

    path = scorer.score()
    # print(scorer.find_distance(path))
    analyser = Analyser()
    contest, legs = analyser.analyse(lat, lon, alt, time, raw_time, path)

    gen = EmissionGenerator(lon, lat, alt, time, raw_time, sensor)
    result = gen.run()

test_pipeline()
