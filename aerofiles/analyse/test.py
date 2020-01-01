from aerofiles.score import Scorer as OLCScorer
from .stats import Analyser as Analyser
from .emissions import EmissionGenerator
from .manager import FlightManager

def test_pipeline():
    manager = FlightManager()
    manager.import_flight('simple_thermal_moritz')
    manager.set_up()
    manager.run()

test_pipeline()
