from .manager import FlightManager

def test_pipeline():
    manager = FlightManager()
    manager.import_flight('simple_thermal_moritz')
    manager.set_up()
    manager.run_contest()

test_pipeline()
